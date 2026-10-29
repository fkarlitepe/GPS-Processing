"""Static/kinematic experimental GNSS AR filter (GPS L1/L2, Galileo E1/E5a) on the shared EKF core
(processing.engine). Fixed estimates never feed the float filter unless fix-and-hold is selected.

Each satellite system has its own reference satellite (ambiguity datum), its own receiver
clocks/biases (DCM: code, phase, MW; IPC: code, phase; SDBS: removed by between-satellite
differences inside each system) and its own wavelengths; the WL and N1 integers of both
systems are searched together, since every candidate is a within-system difference.

The EKF estimators use process noise, a zenith wet delay state, Saastamoinen/relativity
corrections and robust (IGG-III) updates; ``integer_method`` selects LAMBDA/ILS, integer
bootstrapping or TCAR/CIR-style cascade rounding (all with partial AR except 'ils'); the
acceptance policy (P0 ordering, FFRT, satellite exclusion, ML) is ``ar.policy``.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from . import sdbs, dcm, ipc
from .models import LAMBDA_W
from .policy import run_stage
from gps_processing.processing.engine import FilterEngine, Layout
from gps_processing.processing.estimation import FilterOptions
from gps_processing.constants import C
from gps_processing import gnss

CLOCKS = {'SDBS': 0, 'DCM': 3, 'IPC': 2}


def ar_settings(config):
    """AR policy from the configuration; objects without the v2.9 fields (older scripts, tests)
    keep the v2.8 behaviour: ratio test, at least two candidates, no exclusion, no arc/elevation gate."""
    get = lambda name, default: getattr(config, name, default)
    return SimpleNamespace(method=get('integer_method', 'ils'), ratio=config.ar_ratio, max_std=config.ar_max_std,
                           success=get('ar_success_rate', .999), validation=get('ar_validation', 'ratio'),
                           threshold=get('ml_threshold', .99), failure_rate=get('ar_failure_rate', .001),
                           min_ambiguities=get('ar_min_ambiguities', 2), max_exclusions=get('ar_max_exclusions', 0),
                           min_elevation=get('ar_min_elevation', 0.0), min_arc=get('ar_min_arc', 0.0),
                           hold=get('ar_hold', False), hold_epochs=get('ar_hold_epochs', 10),
                           gates=get('ar_candidate_gates', False), fixability=None,
                           fixability_threshold=get('ar_fixability_threshold', 0.5))


def system_satellites(system):
    info = gnss.SYSTEMS[system]
    return [info['offset']+prn for prn in range(1, info['count']+1)]


def redatum(kf, layout, new):
    """Re-reference the (N1, NW) pairs of ``new``'s system to satellite ``new``: x_j -= x_new."""
    t = np.eye(len(kf.x)); k = layout.amb(new)
    for sat in system_satellites(gnss.system_of(new)):
        j = layout.amb(sat)
        t[j, k] -= 1; t[j+1, k+1] -= 1
    kf.x = t@kf.x; kf.P = t@kf.P@t.T


def noise_blocks(sine, code, phase, sats):
    """3x3 (IF code, IF phase, MW) covariance per satellite from raw code/phase sigmas
    (``models.combination_covariance``, vectorized)."""
    out = np.empty((len(sats), 3, 3))
    for system in gnss.SYSTEMS:
        rows = gnss.systems_of(sats) == system
        if not rows.any():
            continue
        k = gnss.constants(system); f1, f2 = k['f1'], k['f2']
        t = np.array([[f1**2/(f1**2-f2**2), -f2**2/(f1**2-f2**2), 0, 0], [0, 0, f1**2/(f1**2-f2**2), -f2**2/(f1**2-f2**2)],
                      [-f1/(f1+f2), -f2/(f1+f2), f1/(f1-f2), -f2/(f1-f2)]])
        # Same operation order as T @ diag(code^2, code^2, phase^2, phase^2) @ T^T / sine^2: the start-up
        # (1e10 clock variances) is ill-conditioned enough to amplify last-digit differences.
        d = np.column_stack([code[rows]**2, code[rows]**2, phase[rows]**2, phase[rows]**2])
        out[rows] = ((t[None, :, :]*d[:, None, :])@t.T)/(sine[rows]**2)[:, None, None]
    return out


class UserModel:
    """Observation plug-in: IF code, IF phase and MW per satellite, (N1, NW) states, one of the
    SDBS / DCM / IPC transformations, reference satellites per system and the AR stages."""

    def __init__(self, config, layout, method, settings, variance_model, validator):
        self.config, self.layout, self.method, self.settings = config, layout, method, settings
        self.transform = {'SDBS': sdbs, 'DCM': dcm, 'IPC': ipc}[method].transform
        self.variance_model, self.validator = variance_model, validator
        self.resolvers = ({system: ipc.WideLaneResolver(config.ar_min_epochs, lambda_w=gnss.constants(system)['lambda_w'])
                           for system in gnss.SYSTEMS} if method == 'IPC' else None)
        self.wide_lane, self.references, self.age, self.arc_start = {}, {}, {}, {}
        self.held = {}                       # fix-and-hold: state index -> [integer, consecutive epochs]
        self.usable = {}                     # sat -> product covers it (``ar_usable`` column, FCB gaps)
        self.positions, self.ambiguities = [], []

    def min_satellites(self, frame):
        return 5

    def reset_arc(self, engine, sat, epoch, frame):
        k = self.layout.amb(sat)
        engine.kf.reset([k, k+1], 0.0, 1e6)
        self.age[sat] = 0; self.arc_start[sat] = epoch
        self.held.pop(k, None); self.held.pop(k+1, None)
        self.wide_lane.pop(sat, None)
        if self.resolvers is not None:
            self.resolvers[gnss.system_of(sat)].reset(sat)

    def prepare(self, engine, epoch, frame, slipped):
        kf, layout, wide_lane = engine.kf, self.layout, self.wide_lane
        ids = frame.sat.to_numpy(dtype=int)
        usable = frame.ar_usable.to_numpy(bool) if 'ar_usable' in frame else np.ones(len(ids), bool)
        slipped = list(slipped)
        for sat, ok in zip(ids, usable):
            # The product correction appears or disappears: a fractional jump of the ambiguity.
            if sat in self.usable and self.usable[sat] != ok and sat not in slipped:
                self.reset_arc(engine, int(sat), epoch, frame)
                slipped.append(int(sat))
            self.usable[sat] = bool(ok)
        frame['ar_usable'] = usable
        present = gnss.systems_of(ids)
        for system in gnss.SYSTEMS:
            mine = ids[present == system]
            reference = self.references.get(system)
            if not len(mine):
                continue
            covered = mine[usable[present == system]]
            if reference not in mine or reference in slipped or (len(covered) and reference not in covered):
                sub_frame = frame[frame.sat.isin(covered if len(covered) else mine)]
                new = int(sub_frame.loc[sub_frame.elevation.idxmax(), 'sat'])
                own = [s for s in wide_lane if gnss.system_of(s) == system]
                if reference is not None and self.age.get(new, 0) > 0 and new not in slipped:
                    # The datum change is an integer shift of the system's ambiguities (absorbed by its
                    # receiver clocks, Shi & Gao 2014): N_j <- N_j - N_new keeps converged and fixed arcs.
                    redatum(kf, layout, new)
                    if new in wide_lane:
                        shift = wide_lane[new]
                        for s in own: wide_lane[s] -= shift
                        wide_lane.pop(new)
                    elif own:
                        # IPC: the new reference has no integer WL yet; re-resolve against it.
                        for s in own: kf.reset(layout.amb(s)+1, 0.0, 1e6); wide_lane.pop(s)
                else:
                    every = system_satellites(system)
                    kf.reset(np.ravel([[layout.amb(s), layout.amb(s)+1] for s in every]), 0.0, 1e6)
                    for s in every: self.age.pop(s, None); wide_lane.pop(s, None)
                self.held.clear()                            # held integers refer to the old datum
                self.references[system] = new
            k = layout.amb(self.references[system])
            kf.reset([k, k+1], 0.0, 0.0)
        for sat in ids:
            if self.age.get(sat, 0) == 0: self.arc_start[sat] = epoch
            self.age[sat] = self.age.get(sat, 0)+1
        frame['arc_epochs'] = [self.age[s] for s in ids]

    def measurements(self, engine, epoch, frame):
        layout, kf = self.layout, engine.kf
        ids = frame.sat.to_numpy(dtype=int)
        n = len(frame)
        y = np.empty(3*n)
        y[0::3] = frame.Pc.to_numpy()+frame.code_m.to_numpy()
        y[1::3] = frame.Lc.to_numpy()+frame.phase_m.to_numpy()
        y[2::3] = frame.A4.to_numpy()+frame.mw_m.to_numpy()
        if self.resolvers is not None:
            for sat, mw in zip(ids, y[2::3]):
                self.resolvers[gnss.system_of(sat)].add(int(sat), mw)
        options = engine.options
        if self.variance_model is None:
            sine = np.maximum(.1, np.sin(np.radians(frame.elevation.to_numpy())))
            blocks = noise_blocks(sine, np.full(n, options.code_sigma), np.full(n, options.phase_sigma), ids)
        else:
            code, phase = self.variance_model.sigmas(frame)
            blocks = noise_blocks(np.ones(n), np.asarray(code, float), np.asarray(phase, float), ids)
        noise = np.zeros((3*n, 3*n))
        index = 3*np.arange(n)[:, None]+np.arange(3)[None, :]
        noise[index[:, :, None], index[:, None, :]] = blocks
        coords, delays, relativity = engine.satellite_terms(epoch, frame)
        wet = delays.wet
        # Satellite-specific delays common to code and phase (MW is geometry free).
        common = delays.common-relativity
        position = {s: i for i, s in enumerate(ids)}
        present = gnss.systems_of(ids)
        ref_pos = np.array([position[self.references[gnss.system_of(s)]] for s in ids])    # per-satellite datum position
        if self.resolvers is not None:
            # IPC wide lane: rounded with the satellite-averaged receiver bias of each system, then held as known.
            for system, res in self.resolvers.items():
                mine = ids[present == system]
                if not len(mine): continue
                undifferenced = res.resolve(mine); ref = self.references[system]
                if ref in undifferenced:
                    for sat, value in undifferenced.items():
                        if sat != ref and sat not in self.wide_lane:
                            self.wide_lane[sat] = value-undifferenced[ref]
                            kf.reset(layout.amb(sat)+1, float(self.wide_lane[sat]), 1e-10)
        is_reference = ref_pos == np.arange(n)
        groups = {'SDBS': np.repeat(np.flatnonzero(~is_reference), 3), 'DCM': np.repeat(np.arange(n), 3),
                  'IPC': np.repeat(np.arange(n), 2)}[self.method]
        per_satellite = 2 if self.method == 'IPC' else 3
        # Variance classes (adaptive EKF): code / phase / MW per system.
        classes = (ids[groups] > 100).astype(int)*3+np.tile(np.arange(per_satellite), len(groups)//per_satellite)
        k = np.array([layout.amb(s) for s in ids], int)
        lambda_n, beta, lambda_w = (gnss.per_row(ids, key) for key in ('lambda_n', 'beta', 'lambda_w'))
        clock_columns = [np.array([layout.clock_of[gnss.system_of(s)][offset] for s in ids], int) for offset in range(layout.clocks)]
        tropo, dim, transform = layout.tropo, layout.dim, self.transform
        code_row, phase_row, mw_row = 3*np.arange(n), 3*np.arange(n)+1, 3*np.arange(n)+2

        def equations(current, inflation=np.ones(n)):
            delta = current[:3]-coords; ranges = np.linalg.norm(delta, axis=1)
            unit = delta/ranges[:, None]
            h = np.zeros((3*n, dim)); predicted = np.zeros(3*n)
            h[code_row, :3] = unit; h[phase_row, :3] = unit
            h[phase_row, k] = lambda_n; h[phase_row, k+1] = beta
            h[mw_row, k+1] = lambda_w
            predicted[code_row] = ranges+common+delays.code
            predicted[phase_row] = ranges+common+delays.phase+lambda_n*current[k]+beta*current[k+1]
            predicted[mw_row] = lambda_w*current[k+1]
            if tropo is not None:
                h[code_row, tropo] = wet; h[phase_row, tropo] = wet
                predicted[code_row] += wet*current[tropo]; predicted[phase_row] += wet*current[tropo]
            for offset, columns in enumerate(clock_columns):
                h[3*np.arange(n)+offset, columns] = 1
                predicted[3*np.arange(n)+offset] += current[columns]
            weight = np.repeat(inflation, 3)**.5
            scaled = noise*weight[:, None]*weight[None, :]
            return (*transform(y, predicted, h, scaled, ref_pos), groups)

        self.delays = delays
        return equations, n, classes

    def hold(self, kf, fixed, narrow):
        """Fix-and-hold (RTKLIB, PRIDE): integers fixed identically for ``hold_epochs`` epochs are fed
        back to the float filter as tight constraints (sigma 0.001 cycle)."""
        for i in narrow:
            for k in (i, i+1):
                value = fixed[k]
                if abs(value-np.rint(value)) > 1e-6: continue
                previous = self.held.get(k)
                same = previous and previous[0] == int(np.rint(value))
                self.held[k] = [int(np.rint(value)), previous[1]+1 if same else 1]
        ready = [k for k, (value, count) in self.held.items() if count >= self.settings.hold_epochs]
        if ready:
            kf.constrain(ready, [self.held[k][0] for k in ready], 1e-6)

    def fix(self, kf, epoch, frame, equations):
        """WL then N1 stage on the candidate arcs; returns (fixed state, status, reason, wl, nl, chosen)."""
        layout, settings, age = self.layout, self.settings, self.age
        state, cov = kf.x, kf.P
        ids = frame.sat.to_numpy(dtype=int)
        datum = lambda s: self.references[gnss.system_of(s)]
        elevation_of = dict(zip(ids, frame.elevation))
        ready = lambda s: age[s] >= self.config.ar_min_epochs and epoch-self.arc_start.get(s, epoch) >= settings.min_arc
        # Candidate arcs: long enough (epochs and duration, PRIDE 15 min) and high enough (15 deg).
        usable = dict(zip(ids, frame.ar_usable))
        candidates = [int(s) for s in ids if s != datum(s) and ready(s) and ready(datum(s)) and elevation_of[s] >= settings.min_elevation
                      and usable[s] and usable[datum(s)] and (self.resolvers is None or s in self.wide_lane)]
        candidates.sort(key=lambda s: cov[layout.amb(s), layout.amb(s)])
        chosen = candidates[:4] if settings.method == 'ils' else candidates
        fixed, status, wl, nl = state.copy(), 'float', None, None
        if len(chosen) < 2:
            return fixed, status, 'arc_or_precision_gate', wl, nl, chosen
        try:
            sat_of = {layout.amb(s): s for s in chosen}
            context = lambda used: (np.array([sat_of[i] for i in used]), np.array([elevation_of[sat_of[i]] for i in used]),
                                    np.array([age[sat_of[i]] for i in used]))
            wide = [layout.amb(s)+1 for s in chosen]
            stage, wl, reason, used = run_stage(state, cov, wide, settings, self.validator, 'wl', context([i-1 for i in wide]))
            if stage is None:
                return fixed, status, reason, wl, nl, chosen
            narrow = [i-1 for i in used]
            stage, nl, reason, used_n1 = run_stage(*stage, narrow, settings, self.validator, 'n1', context(narrow))
            if stage is None:
                return fixed, status, reason, wl, nl, chosen
            candidate = stage[0]
            measured, predicted, _, rr, _ = equations(candidate)
            residual = measured-predicted
            # Integrity check of the fixed solution (not an acceptance test of the integers).
            if float(residual@np.linalg.solve(rr, residual)/len(residual)) > 9:
                return fixed, status, 'residual_gate', wl, nl, chosen
            full = wl['dimension'] == nl['dimension'] == len(candidates)
            if settings.hold:
                self.hold(kf, candidate, used_n1)
            return candidate, 'fixed' if full else 'partial_fixed', 'wl_n1_accepted', wl, nl, chosen
        except (RuntimeError, np.linalg.LinAlgError):
            return fixed, status, 'search_incomplete_or_singular', wl, nl, chosen

    def record(self, engine, epoch, frame, slipped, equations, report):
        kf, layout, delays, method = engine.kf, self.layout, self.delays, self.method
        state, cov = kf.x, kf.P
        fixed, status, reason, wl, nl, chosen = self.fix(kf, epoch, frame, equations)
        ids = frame.sat.to_numpy(dtype=int)
        n = len(frame)
        present = gnss.systems_of(ids)
        datum = lambda s: self.references[gnss.system_of(s)]
        reference = self.references.get('G', self.references.get('E'))
        measured, predicted, _, _, _ = equations(fixed)
        residual = measured-predicted
        rms = float(np.sqrt(np.mean(residual**2)))
        sigma = np.sqrt(np.diag(cov)[:3])
        clocks = layout.clock_of['G']
        resolver = self.resolvers['G'] if self.resolvers else None
        tropo = layout.tropo
        self.positions.append(dict(
            epok=epoch, x=fixed[0], y=fixed[1], z=fixed[2], float_x=state[0], float_y=state[1], float_z=state[2],
            sigma_x=sigma[0], sigma_y=sigma[1], sigma_z=sigma[2],
            receiver_clock_s=state[clocks[0]]/C if clocks else np.nan,
            receiver_phase_clock_s=state[clocks[1]]/C if clocks else np.nan,
            receiver_mw_bias_m=state[clocks[2]] if method == 'DCM' else (resolver.receiver_bias*LAMBDA_W if resolver is not None else np.nan),
            reference_sat_E=self.references.get('E', np.nan) if 'G' in self.references else np.nan,
            gps_satellites=int((present == 'G').sum()), galileo_satellites=int((present == 'E').sum()),
            zwd_m=delays.zwd_apriori+state[tropo] if tropo is not None else (delays.zwd_apriori if engine.atmosphere.tro is not None else np.nan),
            stec_tecu=float(np.nanmean(delays.stec)) if np.isfinite(delays.stec).any() else np.nan,
            satellites=n, status=status, reference_sat=reference, residual_rms_m=rms, nis=report.nis, alpha=report.alpha,
            downweighted=report.downweighted, **{f'variance_factor_{k}': v for k, v in report.factors.items()}))
        integer_value = lambda value: value if status != 'float' and abs(value-np.rint(value)) < 1e-6 else np.nan
        for i, sat in enumerate(ids):
            k = layout.amb(sat)
            self.ambiguities.append(dict(
                epok=epoch, sat=sat, reference_sat=datum(sat), N1_float=state[k], NW_float=state[k+1],
                sigma_N1=float(np.sqrt(max(cov[k, k], 0))), sigma_NW=float(np.sqrt(max(cov[k+1, k+1], 0))), cov_N1_NW=float(cov[k, k+1]),
                arc_epochs=int(self.age[sat]), elevation=float(frame.elevation.iloc[i]), ar_usable=bool(frame.ar_usable.iloc[i]),
                N1_fixed=integer_value(fixed[k]) if sat != datum(sat) else np.nan,
                NW_fixed=integer_value(fixed[k+1]) if sat != datum(sat) else np.nan))
        info = lambda r, key: r.get(key, np.nan) if r else np.nan
        engine.diagnostics.append(dict(
            epok=epoch, status=status, reason=reason, satellites=n, reference_sat=reference,
            wl_ratio=info(wl, 'ratio'), n1_ratio=info(nl, 'ratio'), wl_success=info(wl, 'success'), n1_success=info(nl, 'success'),
            wl_success_upper=info(wl, 'success_upper'), n1_success_upper=info(nl, 'success_upper'), wl_adop=info(wl, 'adop'),
            n1_adop=info(nl, 'adop'), wl_probability=info(wl, 'probability'), n1_probability=info(nl, 'probability'),
            wl_ffrt=info(wl, 'ffrt_threshold'), n1_ffrt=info(nl, 'ffrt_threshold'), n1_min_p0=info(nl, 'min_p0'),
            candidate_pairs=len(chosen), fixed_pairs=nl['dimension'] if status != 'float' else 0,
            wl_known=len(self.wide_lane) if self.resolvers is not None else np.nan,
            reset_satellites=','.join(map(str, slipped)), residual_rms_m=rms, nis=report.nis, downweighted=report.downweighted))


def solve(obs, satellites, approximate_xyz, config, variance_model=None, validator=None, atmosphere=None, fixability=None):
    options = FilterOptions.from_config(config); settings = ar_settings(config)
    settings.fixability = fixability
    if 'ml' in settings.validation and validator is None:
        from gps_processing.processing.ml.ar_validation import default_validator
        validator = default_validator()
    method = config.technique.split('/')[-1]
    layout = Layout(clocks=CLOCKS[method], troposphere=options.estimate_troposphere, per_satellite=2)
    engine = FilterEngine(obs, satellites, approximate_xyz, config, layout, atmosphere, options)
    model = UserModel(config, layout, method, settings, variance_model, validator)
    engine.run(model)
    if not model.positions: raise ValueError('PPP-AR için çözülebilir epok yok.')
    return pd.DataFrame(model.positions), pd.DataFrame(model.ambiguities), pd.DataFrame(engine.diagnostics), engine.kf.P, engine.correction
