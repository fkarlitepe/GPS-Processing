"""Satellite and receiver UFO (FCB / UPD) estimation from float zero-difference ambiguities.

Observation model (Li & Zhang 2012, eqs. 10-12), in cycles, for station i and satellite j:
    b_ij = n_ij + r_i + s_j + e_ij
n_ij integer, r_i receiver UFO, s_j satellite UFO. One receiver UFO is the datum (r = 0);
everything else is defined modulo 1 and the integers absorb the rest.

Three estimators share the table interface (columns station, sat, block, epok, value, sigma
and the optional features elevation, arc_epochs):

* ``batch``  - Li & Zhang (2012): per block, an integrated least-squares adjustment of r and s
               with the integers fixed sequentially (bootstrap from the datum station, fix the
               ambiguities whose fractional residual is below ``gate``, re-adjust, repeat).
* ``ekf``    - the same observation equation in a linear Kalman filter whose state holds every
               r_i and s_j as random walks. Integers are fixed against the predicted state, so
               the UFOs stay continuous across blocks and the datum, and short or sparse
               blocks borrow strength from the previous ones.
* ``ekf+ml`` - two passes of the EKF; between them a gradient-boosted regression, cross-fitted
               by station, learns the residual variance of each float ambiguity from its own
               precision, elevation and arc length, and the random-walk noise of every satellite
               UFO is learned from the first pass (robust variance of its block-to-block changes).
               The second pass uses these variances and leaves out ambiguities predicted to be
               unreliable.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from gps_processing.processing.ml.learners import GradientBoostingRegressor
from gps_processing.processing.ml.stochastic import LOG_CHI2_BIAS

METHODS = ('batch', 'ekf', 'ekf+ml')
MIN_SIGMA = 0.02


def wrap(value):
    """Fraction in [-0.5, 0.5)."""
    return (np.asarray(value, float)+0.5) % 1.0-0.5


def circular_mean(values, weights=None):
    angle = 2*np.pi*np.asarray(values, float)
    weights = np.ones(len(angle)) if weights is None else np.asarray(weights, float)
    return float(np.angle(np.sum(weights*np.exp(1j*angle)))/(2*np.pi))


@dataclass
class UFOResult:
    method: str
    satellite: pd.DataFrame            # block, epok, sat, ufo, sigma
    receiver: pd.DataFrame             # block, epok, station, ufo, sigma
    rows: pd.DataFrame                 # input rows with integer, residual, fixed
    datum: str
    info: dict = field(default_factory=dict)

    def lookup(self, sat, epochs):
        """Satellite UFO at ``epochs`` (block start <= epoch), NaN where not estimated."""
        frame = self.satellite[self.satellite.sat == sat]
        if frame.empty:
            return np.full(len(epochs), np.nan)
        start = frame.epok.to_numpy(float)
        k = np.searchsorted(start, epochs, side='right')-1
        out = np.full(len(epochs), np.nan)
        ok = k >= 0
        out[ok] = frame.ufo.to_numpy(float)[k[ok]]
        return out


def _weights(rows):
    return 1.0/np.maximum(rows.sigma.to_numpy(float), MIN_SIGMA)**2


def _bootstrap(rows, receiver, satellite):
    """Fill unknown r/s by circular means over links to known ones (spanning-tree start)."""
    changed = True
    while changed:
        changed = False
        for sat, group in rows.groupby('sat'):
            if sat in satellite:
                continue
            known = group[group.station.isin(receiver.keys())]
            if len(known):
                satellite[sat] = circular_mean(known.value-known.station.map(receiver), _weights(known))
                changed = True
        for station, group in rows.groupby('station'):
            if station in receiver:
                continue
            known = group[group.sat.isin(satellite.keys())]
            if len(known):
                receiver[station] = circular_mean(known.value-known.sat.map(satellite), _weights(known))
                changed = True


def _adjust_block(rows, datum, gate, iterations=10):
    """Li & Zhang integrated adjustment of one block; returns (r, s, sigma_r, sigma_s, rows)."""
    rows = rows.copy()
    if datum not in set(rows.station):
        datum = rows.station.value_counts().index[0]
    receiver, satellite = {datum: 0.0}, {}
    _bootstrap(rows, receiver, satellite)
    rows = rows[rows.station.isin(receiver.keys()) & rows.sat.isin(satellite.keys())].copy()
    stations = [s for s in receiver if s != datum]
    sats = sorted(satellite)
    column = {('r', s): k for k, s in enumerate(stations)}
    column.update({('s', s): len(stations)+k for k, s in enumerate(sats)})
    x = np.array([receiver[s] for s in stations]+[satellite[s] for s in sats])
    accepted = np.zeros(len(rows), bool)
    covariance = np.full((len(x), len(x)), np.nan)
    station_column = rows.station.map(lambda s: -1 if s == datum else column['r', s]).to_numpy(int)
    sat_column = rows.sat.map(lambda s: column['s', s]).to_numpy(int)
    prediction = lambda x: np.where(station_column >= 0, x[np.maximum(station_column, 0)], 0.0)+x[sat_column]
    for _ in range(iterations):
        predicted = prediction(x)
        integers = np.rint(rows.value.to_numpy()-predicted)
        fraction = rows.value.to_numpy()-predicted-integers
        new = np.abs(fraction) < gate
        if new.sum() < 2:
            break
        design = np.zeros((new.sum(), len(x)))
        chosen = rows[new]
        k = np.arange(new.sum())
        receiver_rows = station_column[new] >= 0
        design[k[receiver_rows], station_column[new][receiver_rows]] = 1
        design[k, sat_column[new]] = 1
        w = _weights(chosen)
        normal = design.T@(design*w[:, None])
        target = design.T@(w*(chosen.value.to_numpy()-integers[new]))
        solution, *_ = np.linalg.lstsq(normal, target, rcond=None)
        residual = chosen.value.to_numpy()-integers[new]-design@solution
        dof = max(len(residual)-np.linalg.matrix_rank(normal), 1)
        scale = max(float(residual@(w*residual))/dof, 1.0)
        covariance = np.linalg.pinv(normal)*scale
        converged = np.array_equal(new, accepted) and np.allclose(solution, x, atol=1e-6)
        x, accepted = solution, new
        if converged:
            break
    predicted = prediction(x)
    rows['integer'] = np.rint(rows.value.to_numpy()-predicted)
    rows['residual'] = rows.value.to_numpy()-predicted-rows.integer.to_numpy()
    rows['fixed'] = np.abs(rows.residual.to_numpy()) < gate
    sigma = np.sqrt(np.abs(np.diag(covariance))) if np.isfinite(covariance).all() else np.full(len(x), np.nan)
    r = {datum: (0.0, 0.0), **{s: (x[column['r', s]], sigma[column['r', s]]) for s in stations}}
    s = {j: (x[column['s', j]], sigma[column['s', j]]) for j in sats}
    return r, s, rows, datum


def _datum(table):
    return table.station.value_counts().index[0]


def batch(table, gate=0.25, datum=None):
    """Li & Zhang (2012): independent integrated adjustments per block with sequential fixing."""
    datum = datum or _datum(table)
    satellites, receivers, rows = [], [], []
    for block, group in table.groupby('block', sort=True):
        if group.station.nunique() < 2:
            continue
        r, s, fixed, used = _adjust_block(group, datum, gate)
        epok = float(group.epok.min())
        satellites += [dict(block=block, epok=epok, sat=int(j), ufo=wrap(v), sigma=e) for j, (v, e) in s.items()]
        receivers += [dict(block=block, epok=epok, station=i, ufo=v, sigma=e) for i, (v, e) in r.items()]
        rows.append(fixed)
    return UFOResult('batch', pd.DataFrame(satellites), pd.DataFrame(receivers),
                     pd.concat(rows, ignore_index=True) if rows else table.iloc[:0], datum)


def ekf(table, gate=0.25, satellite_noise=None, receiver_noise=None, datum=None, sigma=None, method='ekf'):
    """Random-walk Kalman filter over blocks; ``*_noise`` in cycles^2/s (``satellite_noise`` may be a
    {sat: value} dict, the data-driven noise of ``ekf_ml``), ``sigma`` overrides the row standard
    deviations (ML pass)."""
    datum = datum or _datum(table)
    table = table.copy()
    if sigma is not None:
        table['sigma'] = sigma
    span = float(np.median(np.diff(np.sort(table.epok.unique())))) if table.epok.nunique() > 1 else 900.0
    default = 0.01**2/span
    per_satellite = satellite_noise if isinstance(satellite_noise, dict) else {}
    satellite_noise = default if satellite_noise is None or isinstance(satellite_noise, dict) else satellite_noise
    receiver_noise = (0.02**2/span) if receiver_noise is None else receiver_noise
    stations = sorted(set(table.station))
    sats = sorted(set(table.sat))
    index = {('r', s): k for k, s in enumerate(stations)}
    index.update({('s', s): len(stations)+k for k, s in enumerate(sats)})
    n = len(index)
    x, P = np.zeros(n), np.diag(np.full(n, np.inf))
    P[index['r', datum], index['r', datum]] = 0.0
    noise = np.array([receiver_noise]*len(stations)+[per_satellite.get(j, satellite_noise) for j in sats])
    noise[index['r', datum]] = 0.0
    satellites, receivers, rows, previous = [], [], [], None
    for block, group in table.groupby('block', sort=True):
        epok = float(group.epok.min())
        if previous is not None:
            finite = np.isfinite(np.diag(P))
            P[finite, finite] += noise[finite]*(epok-previous)
        previous = epok
        # Bootstrap states never observed before (or still undetermined) from the known ones.
        known_r = {s: x[index['r', s]] for s in stations if np.isfinite(P[index['r', s], index['r', s]]) and s in set(group.station)}
        known_s = {j: x[index['s', j]] for j in sats if np.isfinite(P[index['s', j], index['s', j]]) and j in set(group.sat)}
        before_r, before_s = set(known_r), set(known_s)
        _bootstrap(group, known_r, known_s)
        for key, store, before in (('r', known_r, before_r), ('s', known_s, before_s)):
            for name in set(store)-before:
                k = index[key, name]
                x[k], P[k, :], P[:, k] = store[name], 0.0, 0.0
                P[k, k] = 0.1**2
        usable = group[group.station.map(lambda s: np.isfinite(P[index['r', s], index['r', s]])) &
                       group.sat.map(lambda j: np.isfinite(P[index['s', j], index['s', j]]))].copy()
        if usable.empty:
            continue
        h = np.zeros((len(usable), n))
        h[np.arange(len(usable)), usable.station.map(lambda s: index['r', s]).to_numpy(int)] = 1
        h[np.arange(len(usable)), usable.sat.map(lambda j: index['s', j]).to_numpy(int)] += 1
        values = usable.value.to_numpy(float)
        sigmas = np.maximum(usable.sigma.to_numpy(float), MIN_SIGMA)
        fixed = np.zeros(len(usable), bool)
        for _ in range(4):                                   # sequential fixing inside the block
            finite = np.isfinite(np.diag(P))
            predicted = h@x
            innovation = values-np.rint(values-predicted)-predicted
            spread = np.sqrt(np.einsum('ij,jk,ik->i', h[:, finite], P[np.ix_(finite, finite)], h[:, finite]))
            new = (np.abs(innovation) < gate) & (spread < 0.2) & ~fixed
            if not new.any():
                break
            H = h[new][:, finite]
            R = np.diag(sigmas[new]**2)
            Pf = P[np.ix_(finite, finite)]
            K = np.linalg.solve(H@Pf@H.T+R, H@Pf).T
            x[finite] += K@innovation[new]
            I_KH = np.eye(finite.sum())-K@H
            P[np.ix_(finite, finite)] = I_KH@Pf@I_KH.T+K@R@K.T
            fixed |= new
        predicted = h@x
        usable['integer'] = np.rint(usable.value.to_numpy()-predicted)
        usable['residual'] = usable.value.to_numpy()-predicted-usable.integer.to_numpy()
        usable['fixed'] = fixed
        rows.append(usable)
        for j in set(group.sat):
            k = index['s', j]
            if np.isfinite(P[k, k]):
                satellites.append(dict(block=block, epok=epok, sat=int(j), ufo=float(x[k]), sigma=float(np.sqrt(P[k, k]))))
        for s in set(group.station):
            k = index['r', s]
            if np.isfinite(P[k, k]):
                receivers.append(dict(block=block, epok=epok, station=s, ufo=float(x[k]), sigma=float(np.sqrt(P[k, k]))))
    return UFOResult(method, pd.DataFrame(satellites), pd.DataFrame(receivers),
                     pd.concat(rows, ignore_index=True) if rows else table.iloc[:0], datum)


def features(rows):
    columns = [np.log(np.maximum(rows.sigma.to_numpy(float), 1e-4))]
    for name, transform in (('elevation', lambda v: np.sin(np.radians(v))), ('arc_epochs', np.log1p)):
        values = rows[name].to_numpy(float) if name in rows else np.full(len(rows), np.nan)
        fill = np.nanmedian(values) if np.isfinite(values).any() else 0.0
        columns.append(transform(np.where(np.isfinite(values), values, fill)))
    return np.column_stack(columns)


def learned_sigma(rows, folds=3, seed=0, floor=0.01):
    """Cross-fitted (by station) GBRT of log squared residuals -> standard deviation per row."""
    target = np.log(wrap(rows.residual.to_numpy(float))**2+floor**2)
    x = features(rows)
    stations = np.array(sorted(set(rows.station)))
    group = rows.station.map({s: k % folds for k, s in enumerate(stations)}).to_numpy()
    prediction = np.empty(len(rows))
    for fold in range(folds):
        test = group == fold
        if not test.any():
            continue
        train = ~test if (~test).sum() >= 40 else np.ones(len(rows), bool)
        model = GradientBoostingRegressor(n_estimators=60, max_depth=2, min_samples_leaf=15, seed=seed).fit(x[train], target[train])
        prediction[test] = model.predict(x[test])
    return np.sqrt(np.exp(prediction+LOG_CHI2_BIAS))


def ekf_ml(table, gate=0.25, reliable=0.2, **kw):
    """EKF -> learned per-ambiguity sigma -> EKF with those weights (unreliable rows left out)."""
    first = ekf(table, gate, **kw)
    rows = first.rows
    if len(rows) < 40:
        first.method, first.info['ml'] = 'ekf+ml', 'too few rows for learning; EKF result returned'
        return first
    sigma = learned_sigma(rows)
    keyed = rows[['station', 'sat', 'block']].assign(learned=sigma)
    merged = table.merge(keyed, on=['station', 'sat', 'block'], how='left')
    learned = merged.learned.to_numpy(float)
    # Rows not used in the first pass keep their own sigma; predicted-unreliable rows are dropped.
    learned = np.where(np.isfinite(learned), learned, merged.sigma.to_numpy(float))
    keep = learned < reliable
    noise = process_noise(first)
    second = ekf(merged[keep].drop(columns='learned'), gate, sigma=learned[keep], method='ekf+ml',
                 **{**kw, 'satellite_noise': noise or kw.get('satellite_noise')})
    second.info.update(ml_rows=int(len(rows)), ml_dropped=int((~keep).sum()), ml_median_sigma=float(np.median(sigma)),
                       learned_process_noise_cycles_per_sqrt_h={str(j): float(np.sqrt(q*3600)) for j, q in sorted(noise.items())})
    return second


def process_noise(result, floor=0.002, ceiling=0.2):
    """Data-driven random-walk variance (cycles^2/s) of every satellite UFO from the first EKF pass:
    robust (MAD) variance of the block-to-block changes minus their formal variance, per second.
    Satellites with fewer than 4 changes keep the default. Bounded to [floor, ceiling] cycles per
    block so that one noisy block cannot freeze or free a satellite."""
    out = {}
    for sat, frame in result.satellite.groupby('sat'):
        frame = frame.sort_values('epok')
        if len(frame) < 5:
            continue
        change = wrap(np.diff(frame.ufo.to_numpy(float)))
        dt = np.diff(frame.epok.to_numpy(float))
        formal = frame.sigma.to_numpy(float)**2
        spread = (1.4826*np.median(np.abs(change-np.median(change))))**2
        step = float(np.median(dt))
        variance = max(spread-float(np.median(formal[1:]+formal[:-1])), floor**2)
        out[int(sat)] = min(variance, ceiling**2)/step
    return out


ESTIMATORS = {'batch': batch, 'ekf': ekf, 'ekf+ml': ekf_ml}


def estimate(table, method, gate=0.25, **kw):
    if method not in ESTIMATORS:
        raise ValueError(f'FCB method must be one of {", ".join(METHODS)}.')
    return ESTIMATORS[method](table, gate=gate, **kw) if method != 'batch' else batch(table, gate=gate, datum=kw.get('datum'))


def user_residuals(result, rows):
    """User-side check: correct ``rows`` (one station) with the satellite UFOs, remove the
    receiver part per block by a circular mean (as a between-satellite difference does) and
    return the remaining fractions; blocks with fewer than 3 satellites are skipped."""
    out = []
    for block, group in rows.groupby('block'):
        corrected = group.value.to_numpy(float)-np.array([result.lookup(int(j), np.array([e]))[0]
                                                         for j, e in zip(group.sat, group.epok)]) if len(group) else np.array([])
        ok = np.isfinite(corrected)
        if ok.sum() < 3:
            continue
        receiver = circular_mean(corrected[ok], 1/np.maximum(group.sigma.to_numpy(float)[ok], MIN_SIGMA)**2)
        out.append(wrap(corrected[ok]-receiver))
    return np.concatenate(out) if out else np.array([])


def leave_one_out(table, method, gate=0.25, max_stations=20, **kw):
    """Estimate without each station in turn and test the UFOs on that station's ambiguities.
    Large networks (100+ stations) validate an evenly spread subset of ``max_stations``."""
    rows = []
    stations = sorted(set(table.station))
    if max_stations and len(stations) > max_stations:
        stations = [stations[int(k)] for k in np.linspace(0, len(stations)-1, max_stations)]
    for station in stations:
        rest = table[table.station != station]
        if rest.station.nunique() < 2:
            continue
        result = estimate(rest, method, gate, **kw)
        fractions = user_residuals(result, table[table.station == station])
        if len(fractions):
            rows.append(dict(method=method, station=station, ambiguities=len(fractions),
                             rms_cycles=float(np.sqrt(np.mean(fractions**2))),
                             within_015=float(np.mean(np.abs(fractions) < 0.15)),
                             within_025=float(np.mean(np.abs(fractions) < 0.25))))
    return pd.DataFrame(rows)
