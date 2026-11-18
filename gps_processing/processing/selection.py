"""Satellite selection for PPP and the dilution-of-precision (DOP / weighted DOP) analysis.

Every epoch, after the elevation mask and the cycle-slip check (which always see every visible
satellite), the selector may keep only ``count`` satellites for the filter update. The subset is
chosen by minimising a (weighted) DOP of the PPP geometry:

    G = [ -e_i (ENU) | clock column of the satellite's system | m_w(el_i) (if ZWD is estimated) ]
    Q = (G^T W G)^-1,   W = diag(w_i)
    GDOP = sqrt(tr Q),  PDOP = sqrt(Q_EE + Q_NN + Q_UU),  HDOP = sqrt(Q_EE + Q_NN),  VDOP = sqrt(Q_UU)

With ``weighting='none'`` (W = I) these are the classical DOPs; otherwise they are the weighted
DOPs (WDOP): ``elevation`` (w = sin^2 el, the filter's elevation model), ``cn0`` (sigma^2 =
a 10^(-C/N0/10), w = 1 at 45 dB-Hz) or ``filter`` (w = 1 / sigma_code^2 of the filter's own
stochastic model, IF factor included: the geometry the EKF actually sees; unit weight = a GPS
satellite at the zenith, so weighted and classical DOPs share their scale).

Methods: ``all`` (no selection, analysis only), ``top-cn0``, ``az-sector``, ``random``, greedy
``backward`` elimination, ``fibo-gnss`` (Fibonacci-sphere candidate generation, best of 48 point
sets; Karlitepe, this project) and their ``+ls`` variants (1-swap local search) and
``exhaustive`` (full enumeration, small sets only).

The analysis tables written for each epoch and visible satellite are described in
``SatelliteSelector.select`` and ``record_update``.
"""
from dataclasses import dataclass
from itertools import combinations
import time
import numpy as np
import pandas as pd
from gps_processing import gnss

METHODS = ('all', 'top-cn0', 'az-sector', 'random', 'backward', 'backward+ls', 'fibo-gnss', 'fibo-gnss+ls', 'exhaustive')
WEIGHTINGS = ('none', 'elevation', 'cn0', 'filter')
OBJECTIVES = ('pdop', 'gdop', 'hdop', 'vdop')
GOLDEN_ANGLE = np.pi*(3.0-np.sqrt(5.0))
SYSTEM_INDEX = {'G': 0, 'E': 1}
EXHAUSTIVE_LIMIT = 300_000          # subsets; above it 'exhaustive' falls back to fibo-gnss+ls


@dataclass
class SelectionOptions:
    method: str = 'all'
    count: int = 0                  # satellites kept per epoch; 0 or >= visible: all
    weighting: str = 'filter'       # DOP weights used by the selection
    objective: str = 'pdop'
    interval: float = 0.0           # s; keep the previous subset this long while it stays visible
    hysteresis: float = 0.0         # switch to a new subset only if it lowers the objective by this fraction
    troposphere: bool = True        # include the ZWD column in the DOP geometry when ZWD is estimated
    analysis: bool = False          # write the per-epoch / per-satellite analysis tables
    seed: int = 0

    @classmethod
    def from_config(cls, config):
        return cls(getattr(config, 'selection', 'all'), int(getattr(config, 'selection_count', 0)),
                   getattr(config, 'selection_weighting', 'filter'), getattr(config, 'selection_objective', 'pdop'),
                   float(getattr(config, 'selection_interval', 0.0)), float(getattr(config, 'selection_hysteresis', 0.0)),
                   bool(getattr(config, 'selection_troposphere', True)),
                   bool(getattr(config, 'selection_analysis', False)) or getattr(config, 'selection', 'all') != 'all',
                   int(getattr(config, 'selection_seed', 0)))

    @property
    def active(self):
        return self.method != 'all' and self.count > 0

    def validate(self):
        if self.method not in METHODS:
            raise ValueError(f'Uydu seçim yöntemi {"/".join(METHODS)} olmalı.')
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f'Seçim ağırlığı {"/".join(WEIGHTINGS)} olmalı.')
        if self.objective not in OBJECTIVES:
            raise ValueError(f'Seçim ölçütü {"/".join(OBJECTIVES)} olmalı.')
        if self.method != 'all' and not 6 <= self.count <= 40:
            raise ValueError('Seçilecek uydu sayısı 6 ile 40 arasında olmalı (GPS+Galileo PPP en az 6 bilinmeyen içerir).')
        if self.interval < 0 or not 0 <= self.hysteresis < 1:
            raise ValueError('Seçim aralığı negatif olamaz; histerezis 0 ile 1 arasında olmalı.')


# ------------------------------------------------------------------ geometry and DOP
def enu_rotation(xyz):
    x, y, z = xyz
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y)*(1-6.69437999014e-3))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl*co, -sl*so, cl], [cl*co, cl*so, sl]])


def line_of_sight(receiver, coords):
    """Unit receiver->satellite vectors in ENU, elevation and azimuth (rad)."""
    delta = np.asarray(coords, float)-np.asarray(receiver, float)
    e = (delta/np.linalg.norm(delta, axis=1)[:, None])@enu_rotation(receiver).T
    return e, np.arcsin(np.clip(e[:, 2], -1, 1)), np.arctan2(e[:, 0], e[:, 1]) % (2*np.pi)


def wet_mapping(el):
    """Chao-type approximation of the wet mapping function (enough for the DOP geometry)."""
    s = np.sin(el)
    return 1.0/(s+0.00035/(np.tan(el)+0.017))


class Geometry:
    """Design rows of the visible satellites and batched DOP evaluation of many subsets."""

    def __init__(self, e, systems, weights, wet=None):
        n = len(e)
        self.n, self.systems, self.w = n, np.asarray(systems), np.asarray(weights, float)
        nsys = int(self.systems.max())+1 if n else 1
        cols = 3+nsys+(wet is not None)
        g = np.zeros((n, cols))
        g[:, :3] = -e
        g[np.arange(n), 3+self.systems] = 1.0
        if wet is not None:
            g[:, -1] = wet
        self.g, self.outer = g, np.einsum('ni,nj->nij', g, g)
        self.clock_cols = np.arange(3, 3+nsys)

    def q_matrices(self, masks, weights=None):
        """Cofactor matrices of each subset; absent clock columns are decoupled (unit diagonal)."""
        masks = np.atleast_2d(masks).astype(float)
        w = self.w if weights is None else weights
        N = np.einsum('mn,nij->mij', masks*w, self.outer)
        p = N.shape[1]
        absent = np.zeros((len(masks), p), bool)
        absent[:, self.clock_cols] = np.einsum('mii->mi', N)[:, self.clock_cols] < 1e-12
        N[:, np.arange(p), np.arange(p)] += absent
        ok = np.ones(len(masks), bool)
        try:
            cond = np.linalg.cond(N)
            ok = np.isfinite(cond) & (cond < 1e10)
            Q = np.full_like(N, np.nan)
            if ok.any():
                Q[ok] = np.linalg.inv(N[ok])
        except np.linalg.LinAlgError:
            Q = np.full_like(N, np.nan)
            for i in range(len(N)):
                try:
                    Q[i] = np.linalg.inv(N[i])
                except np.linalg.LinAlgError:
                    ok[i] = False
        return Q, absent, ok

    def dops(self, masks, weights=None):
        """dict of GDOP/PDOP/HDOP/VDOP arrays (inf where the position is not estimable)."""
        Q, absent, ok = self.q_matrices(masks, weights)
        d = np.einsum('mii->mi', Q)
        d = np.where(absent, 0.0, d)
        out = {}
        for name, value in (('gdop', d.sum(1)), ('pdop', d[:, :3].sum(1)), ('hdop', d[:, :2].sum(1)), ('vdop', d[:, 2])):
            v = np.full(len(d), np.inf)
            good = ok & np.isfinite(value) & (value > 0)
            v[good] = np.sqrt(value[good])
            out[name] = v
        return out

    def score(self, masks, objective):
        return self.dops(masks)[objective]


def masks_of(subsets, n):
    m = np.zeros((len(subsets), n), bool)
    for j, s in enumerate(subsets):
        m[j, list(s)] = True
    return m


# ------------------------------------------------------------------ selection algorithms
class Context:
    def __init__(self, geometry, objective, el, az, e, quality, k, rng):
        self.geo, self.objective, self.el, self.az, self.e = geometry, objective, el, az, e
        self.quality, self.k, self.rng, self.n = quality, k, rng, geometry.n
        self.systems = geometry.systems

    def score(self, subsets):
        return self.geo.score(masks_of(subsets, self.n), self.objective)

    def prune(self, sel):
        """A system with one satellite only feeds its own clock: drop it."""
        sel = list(sel)
        counts = np.bincount(self.systems[sel], minlength=2) if sel else np.zeros(2, int)
        return [i for i in sel if counts[self.systems[i]] >= 2]

    def fill(self, sel):
        """Greedy forward addition (minimum objective) until k satellites."""
        sel = list(sel)
        while len(sel) < self.k:
            rest = [i for i in range(self.n) if i not in sel]
            if not rest:
                break
            v = self.score([sel+[i] for i in rest])
            if np.all(~np.isfinite(v)):
                present = set(self.systems[sel].tolist())
                pick = max(rest, key=lambda i: (self.systems[i] in present, self.quality[i]))
            else:
                pick = rest[int(np.argmin(v))]
            sel.append(pick)
        return sel

    def swap(self, sel, max_iter=20):
        """Best-improvement 1-swap local search."""
        sel = list(sel)
        best = self.score([sel])[0]
        for _ in range(max_iter):
            rest = [i for i in range(self.n) if i not in sel]
            pairs = [(a, b) for a in sel for b in rest]
            if not pairs:
                break
            v = self.score([[x for x in sel if x != a]+[b] for a, b in pairs])
            j = int(np.argmin(v))
            if v[j] < best-1e-9:
                best = v[j]
                sel.remove(pairs[j][0])
                sel.append(pairs[j][1])
            else:
                break
        return sel


def select_top_cn0(c):
    return c.fill(c.prune(list(np.argsort(-c.quality)[:c.k])))


def select_random(c):
    sel = list(range(c.k))
    for _ in range(50):
        sel = list(c.rng.choice(c.n, c.k, replace=False))
        if np.isfinite(c.score([sel])[0]):
            break
    return sel


def select_az_sector(c):
    """Highest satellite + the lowest satellite of each of k-1 azimuth sectors (classic quasi-optimal)."""
    top = int(np.argmax(c.el))
    sel = [top]
    width = 2*np.pi/(c.k-1)
    sector = np.floor(((c.az-c.az[top]) % (2*np.pi))/width).astype(int)
    for j in range(c.k-1):
        cand = [i for i in range(c.n) if sector[i] == j and i not in sel]
        if cand:
            sel.append(min(cand, key=lambda i: c.el[i]))
    return c.fill(c.prune(sel))


def select_backward(c):
    sel = list(range(c.n))
    while len(sel) > c.k:
        v = c.score([[x for x in sel if x != i] for i in sel])
        sel.pop(int(np.argmin(v)))
    return sel


def fibonacci_cap(N, rotation=0.0, mask=np.radians(10.0), sign=1.0):
    """N equal-area golden-angle directions (ENU) on the sky above ``mask``."""
    i = np.arange(N)+0.5
    z0 = np.sin(mask)
    z = 1.0-(1.0-z0)*i/N
    az = (sign*np.arange(N)*GOLDEN_ANGLE+rotation) % (2*np.pi)
    r = np.sqrt(1-z*z)
    return np.stack([r*np.sin(az), r*np.cos(az), z], 1)


def select_fibo_gnss(c, rotations=12, lam=0.15, mask=np.radians(10.0)):
    """Fibo-GNSS: (1) pool = best-quality satellites of max(2k, 12) Fibonacci sky cells (keeping those
    within 80 % of the cell's best weight); (2) for N in {k, k+1}, two spiral directions and 12
    rotations, match the N reference directions to pool satellites greedily by angle - lam ln(w/w_max);
    (3) prune single-satellite systems and fill greedily; (4) return the minimum-objective candidate."""
    pool = np.arange(c.n)
    if c.n > c.k+2:
        cells = np.argmax(c.e@fibonacci_cap(max(2*c.k, 12), mask=mask).T, 1)
        keep = []
        for cell in np.unique(cells):
            idx = np.flatnonzero(cells == cell)
            keep += list(idx[c.quality[idx] >= 0.8*c.quality[idx].max()])
        if len(keep) >= c.k:
            pool = np.array(sorted(keep))
    lw = np.log(np.maximum(c.quality[pool], 1e-12)/max(c.quality[pool].max(), 1e-12))
    candidates = set()
    for N in (c.k, c.k+1):
        for sign in (1.0, -1.0):
            for r in range(rotations):
                refs = fibonacci_cap(N, 2*np.pi*r/rotations, mask, sign)
                cost = np.arccos(np.clip(c.e[pool]@refs.T, -1, 1))-lam*lw[:, None]
                used_s, used_r, sel = set(), set(), []
                for flat in np.argsort(cost, axis=None):
                    si, ri = divmod(int(flat), N)
                    if si in used_s or ri in used_r:
                        continue
                    used_s.add(si)
                    used_r.add(ri)
                    sel.append(int(pool[si]))
                    if len(sel) == c.k:
                        break
                candidates.add(tuple(sorted(sel)))
    fixed = list({tuple(sorted(c.fill(c.prune(s)))) for s in candidates})
    return list(fixed[int(np.argmin(c.score(fixed)))])


def select_exhaustive(c):
    total = 1
    for j in range(c.k):
        total = total*(c.n-j)//(j+1)
    if total > EXHAUSTIVE_LIMIT:
        return c.swap(select_fibo_gnss(c))
    best, value = None, np.inf
    it = combinations(range(c.n), c.k)
    while True:
        block = [s for _, s in zip(range(20000), it)]
        if not block:
            break
        v = c.score(block)
        j = int(np.argmin(v))
        if v[j] < value:
            best, value = list(block[j]), v[j]
    return best


ALGORITHMS = {
    'top-cn0': select_top_cn0, 'random': select_random, 'az-sector': select_az_sector,
    'backward': select_backward, 'backward+ls': lambda c: c.swap(select_backward(c)),
    'fibo-gnss': select_fibo_gnss, 'fibo-gnss+ls': lambda c: c.swap(select_fibo_gnss(c)),
    'exhaustive': select_exhaustive,
}


# ------------------------------------------------------------------ the per-epoch selector
def cn0_weight(cn0):
    return 10**((np.asarray(cn0, float)-45.0)/10.0)


class SatelliteSelector:
    """Chooses the satellites of each epoch and keeps the analysis tables.

    ``epochs`` rows (one per epoch): visible / selected counts per system, the objective value,
    GDOP/PDOP/HDOP/VDOP and their weighted counterparts (W*) of all visible satellites and of the
    selected subset, whether the subset was re-selected, and the selection time.

    ``satellites`` rows (one per epoch and visible satellite): elevation, azimuth, C/N0, selection
    weight, ``selected``, and ``delta_dop``: for a selected satellite how much the objective grows
    if it is removed (its geometric value), for an excluded one how much it would change if added
    (negative: it would have improved the DOP). ``record_update`` adds, for the satellites used in
    the update, the post-fit code/phase residuals, the a-priori model terms (troposphere slant
    delay, satellite clock, relativity, second-order ionosphere, ambiguity) and the satellite's own
    share of the Kalman position correction in east/north/up (the shares of all satellites add up
    to the update; the first epochs carry the large initial corrections, so summaries use medians).
    """

    def __init__(self, options, weights_model=None, troposphere=False, mask_deg=10.0):
        options.validate()
        self.options, self.weights_model = options, weights_model
        self.troposphere = troposphere and options.troposphere
        self.mask = np.radians(mask_deg)
        self.rng = np.random.default_rng(options.seed)
        self.previous, self.previous_time = None, None
        self.epochs, self.satellites, self.updates = [], [], []

    def weights(self, frame, el, weighting):
        if weighting == 'none':
            return np.ones(len(frame))
        if weighting == 'elevation':
            return np.maximum(np.sin(el), 0.1)**2
        if weighting == 'cn0':
            return cn0_weight(self.cn0(frame))
        model = self.weights_model
        if model is None:
            return np.maximum(np.sin(el), 0.1)**2
        code, _ = model.sigmas(frame)
        sigma = gnss.per_row(frame.sat.to_numpy(int), 'if_factor')*code
        # Unit weight: a GPS satellite at the zenith (the elevation model), so WDOP keeps the DOP scale.
        zenith = getattr(model, 'code_sigma', None) or getattr(getattr(model, 'fallback', None), 'code_sigma', None)
        reference = zenith*gnss.constants('G')['if_factor'] if zenith else np.median(sigma)
        return (reference/sigma)**2

    @staticmethod
    def cn0(frame):
        for name in ('cn0', 'snr1'):
            if name in frame:
                v = frame[name].to_numpy(dtype=float)
                if np.isfinite(v).any():
                    return np.where(np.isfinite(v), v, np.nanmedian(v))
        return np.full(len(frame), 45.0)

    def select(self, epoch, frame, receiver):
        """Returns the rows of ``frame`` to use in the update (all rows if no selection is active)."""
        t0 = time.perf_counter()
        opt, n = self.options, len(frame)
        sats = frame.sat.to_numpy(int)
        e, el, az = line_of_sight(receiver, frame[['x', 'y', 'z']].to_numpy())
        systems = np.array([SYSTEM_INDEX[s] for s in gnss.systems_of(sats)])
        wet = wet_mapping(el) if self.troposphere else None
        w = self.weights(frame, el, opt.weighting)
        geo = Geometry(e, systems, w, wet)
        quality = cn0_weight(self.cn0(frame)) if opt.weighting in ('none', 'elevation') else w
        k = min(opt.count, n) if opt.active else n
        reselected = False
        if not opt.active or k >= n:
            sel = list(range(n))
        else:
            keep, old = None, None
            idx = {s: i for i, s in enumerate(sats)}
            if self.previous is not None and len(self.previous) == k and all(s in idx for s in self.previous):
                old = [idx[s] for s in self.previous]
                if opt.interval > 0 and epoch-self.previous_time < opt.interval:
                    keep = old
            if keep is None:
                c = Context(geo, opt.objective, el, az, e, quality, k, self.rng)
                keep = ALGORITHMS[opt.method](c)
                if old is not None and opt.hysteresis > 0 and sorted(old) != sorted(keep):
                    # Phase ambiguities converge only while a satellite is used: keep the current subset
                    # unless the new one is clearly better.
                    v_old, v_new = c.score([old, keep])
                    if np.isfinite(v_old) and v_new > v_old*(1-opt.hysteresis):
                        keep = old
                reselected = old is None or sorted(old) != sorted(keep)
                self.previous, self.previous_time = sats[keep].tolist(), epoch
            sel = sorted(keep)
        elapsed = (time.perf_counter()-t0)*1e3
        if opt.analysis:
            self.log(epoch, frame, sats, systems, el, az, e, wet, w, geo, sel, reselected, elapsed)
        return frame.iloc[sel].reset_index(drop=True)

    def log(self, epoch, frame, sats, systems, el, az, e, wet, w, geo, sel, reselected, elapsed):
        n = len(sats)
        chosen = np.zeros(n, bool)
        chosen[sel] = True
        masks = np.vstack([np.ones(n, bool), chosen])
        plain = geo.dops(masks, np.ones(n))
        weighted = geo.dops(masks)
        objective = weighted[self.options.objective] if self.options.weighting != 'none' else plain[self.options.objective]
        row = dict(epok=epoch, visible=n, selected=len(sel), visible_gps=int((systems == 0).sum()),
                   visible_galileo=int((systems == 1).sum()), selected_gps=int((systems[sel] == 0).sum()),
                   selected_galileo=int((systems[sel] == 1).sum()), method=self.options.method,
                   weighting=self.options.weighting, objective=self.options.objective,
                   objective_all=float(objective[0]), objective_selected=float(objective[1]),
                   reselected=reselected, selection_ms=elapsed)
        for name in OBJECTIVES:
            row[f'{name}_all'], row[f'{name}_selected'] = float(plain[name][0]), float(plain[name][1])
            row[f'w{name}_all'], row[f'w{name}_selected'] = float(weighted[name][0]), float(weighted[name][1])
        self.epochs.append(row)
        # Marginal value of each satellite for the selected subset (objective with the chosen weights).
        toggled = np.repeat(chosen[None], n, 0)
        toggled[np.arange(n), np.arange(n)] = ~chosen
        geo_w = np.ones(n) if self.options.weighting == 'none' else None
        changed = geo.dops(toggled, geo_w)[self.options.objective]
        base = float(objective[1])
        self.satellites.append(pd.DataFrame(dict(
            epok=epoch, sat=sats, prn=[gnss.label(s) for s in sats], system=gnss.systems_of(sats),
            elevation=np.degrees(el), azimuth=np.degrees(az), cn0=self.cn0(frame), weight=w, selected=chosen,
            delta_dop=np.where(np.isfinite(changed), changed-base, np.inf))))

    def record_update(self, epoch, frame, receiver, report, terms=None, clocks=()):
        """Per-satellite model terms and Kalman contributions of the satellites in the update.

        The receiver clocks are white noise, so every innovation carries the common clock offset
        (tens of km); it is absorbed by the clock states and cancels in the sum, but would dominate
        each satellite's share. The shares are therefore computed from the innovations reduced by
        the estimated clock update, v_i - h_i,clock dx_clock (their sum equals the position update
        up to the negligible gain of the position on a common clock shift)."""
        if not self.options.analysis or report is None or getattr(report, 'gain', None) is None:
            return
        n = len(frame)
        gain, innovation, groups = report.gain, report.innovation, np.asarray(report.groups)
        rows = len(innovation)
        clocks = list(clocks)
        if clocks and getattr(report, 'design', None) is not None:
            update = gain@innovation
            innovation = innovation-report.design[:, clocks]@update[clocks]
        contribution = np.zeros((n, 3))
        for g in np.unique(groups[:rows]):
            r = np.flatnonzero(groups[:rows] == g)
            if 0 <= g < n:
                contribution[g] = gain[:3, r]@innovation[r]
        enu = contribution@enu_rotation(receiver).T
        table = dict(epok=epoch, sat=frame.sat.to_numpy(int), correction_e_m=enu[:, 0], correction_n_m=enu[:, 1], correction_u_m=enu[:, 2],
                     correction_3d_m=np.linalg.norm(enu, axis=1), inflation=report.inflation[:n] if len(report.inflation) >= n else np.nan)
        for key, value in (terms or {}).items():
            table[key] = value
        self.updates.append(pd.DataFrame(table))

    def tables(self):
        """(epochs, satellites) data frames; satellites joined with the update terms."""
        epochs = pd.DataFrame(self.epochs)
        sats = pd.concat(self.satellites, ignore_index=True) if self.satellites else pd.DataFrame()
        if self.updates and not sats.empty:
            sats = sats.merge(pd.concat(self.updates, ignore_index=True), on=['epok', 'sat'], how='left')
        return epochs, sats


def summarise(epochs, satellites):
    """Per-satellite and overall statistics of a selection run (analysis section)."""
    if epochs is None or epochs.empty:
        return {}, pd.DataFrame()
    overall = dict(epochs=len(epochs), method=str(epochs.method.iloc[0]), weighting=str(epochs.weighting.iloc[0]),
                   objective=str(epochs.objective.iloc[0]), mean_visible=float(epochs.visible.mean()),
                   mean_selected=float(epochs.selected.mean()), reselections=int(epochs.reselected.sum()),
                   mean_selection_ms=float(epochs.selection_ms.mean()))
    for name in OBJECTIVES:
        for prefix in ('', 'w'):
            for which in ('all', 'selected'):
                col = f'{prefix}{name}_{which}'
                v = epochs[col].replace([np.inf, -np.inf], np.nan)
                overall[f'median_{col}'] = float(v.median())
    ratio = (epochs.objective_selected/epochs.objective_all).replace([np.inf, -np.inf], np.nan)
    overall['median_objective_ratio'] = float(ratio.median())
    if satellites is None or satellites.empty:
        return overall, pd.DataFrame()
    s = satellites.copy()
    s['selected'] = s.selected.astype(bool)
    agg = dict(visible_epochs=('epok', 'size'), selected_epochs=('selected', 'sum'), mean_elevation=('elevation', 'mean'),
               mean_cn0=('cn0', 'mean'), mean_weight=('weight', 'mean'))
    per = s.groupby('prn').agg(**agg)
    per['selected_share'] = per.selected_epochs/per.visible_epochs
    used = s[s.selected]
    if 'correction_3d_m' in used:
        extra = dict(median_correction_3d_mm=('correction_3d_m', lambda v: 1e3*np.nanmedian(np.abs(v))))
        if 'code_residual_m' in used:
            extra.update(rms_code_residual_m=('code_residual_m', lambda v: float(np.sqrt(np.nanmean(np.square(v))))),
                         rms_phase_residual_mm=('phase_residual_m', lambda v: 1e3*float(np.sqrt(np.nanmean(np.square(v))))))
        per = per.join(used.groupby('prn').agg(**extra,
                                               median_delta_dop_selected=('delta_dop', lambda v: float(np.nanmedian(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan)))
    excluded = s[~s.selected]
    if len(excluded):
        per = per.join(excluded.groupby('prn').agg(median_delta_dop_excluded=('delta_dop', lambda v: float(np.nanmedian(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan)))
    return overall, per.reset_index().sort_values('prn')
