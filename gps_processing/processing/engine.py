"""One EKF core for float PPP and the PPP-AR user models.

The epoch loop, state layout, elevation mask, prediction, receiver-clock and cycle-slip resets,
atmospheric delays, satellite clock terms and the robust/adaptive update live here once;
an observation model (plug-in) supplies the measurement equations and its own bookkeeping:

* ``processing.ppp_ekf.IonosphereFreeModel``: IF code + phase, one IF ambiguity per satellite;
* ``processing.ar.solver.UserModel``: IF code, IF phase and MW with (N1, NW) per satellite and the
  SDBS / DCM / IPC transformation; integer fixing runs after the float update (fixed
  estimates never feed the filter unless fix-and-hold is selected).

State layout of both:
    [X Y Z | receiver clocks of the first system (0-3) | ZWD | ambiguities (gnss.slots_for(systems) x k) |
     Galileo receiver clocks (same number) | BDS receiver clocks (if BDS is processed)]
"""
import time
import numpy as np
from gps_processing.constants import C
from gps_processing.processing.geometry import elevation
from gps_processing.processing.cycle_slip import SlipDetector
from gps_processing.processing.atmosphere import Atmosphere
from gps_processing.processing.estimation import ExtendedKalmanFilter, FilterOptions
from gps_processing.processing.estimation.quality import QualityOptions
from gps_processing import gnss


class Layout:
    def __init__(self, clocks, troposphere, per_satellite, systems=('G', 'E')):
        self.clocks, self.per_satellite = clocks, per_satellite
        base = 3+clocks
        self.tropo = base if troposphere else None
        self.first = base+bool(troposphere)
        self.galileo = self.first+per_satellite*gnss.slots_for(systems)
        self.dim = self.galileo+clocks
        self.clock_of = {'G': list(range(3, base)), 'E': list(range(self.galileo, self.dim))}
        if 'C' in systems:                     # BDS receiver clock (B1I/B3I IF code datum) after the Galileo clock
            self.clock_of['C'] = list(range(self.dim, self.dim+clocks))
            self.dim += clocks
        self.all_clocks = [i for s in self.clock_of for i in self.clock_of[s]]

    def amb(self, sat):
        """First ambiguity state of ``sat`` (AR: N1, NW follows)."""
        return self.first+self.per_satellite*gnss.slot(sat)


class FilterEngine:
    def __init__(self, observations, satellites, approximate_xyz, config, layout, atmosphere=None, options=None):
        self.config, self.layout = config, layout
        self.options = options or FilterOptions.from_config(config)
        self.quality = QualityOptions.from_config(config)    # outlier tests and robust weight function
        self.mask = getattr(config, 'elevation_mask', 10)
        columns = ['epok', 'sat', 'x', 'y', 'z']+(['relativity_m'] if 'relativity_m' in satellites else [])
        self.data = observations.merge(satellites[columns], on=['epok', 'sat'], validate='one_to_one')
        self.atmosphere = atmosphere or Atmosphere(approximate_xyz, physical=self.options.physical_corrections)
        # hold_coordinates: position fixed to the reference (NL diagnostics, FCB network stations).
        self.coordinates_held = getattr(config, 'hold_coordinates', False) and getattr(config, 'reference_xyz', None) is not None
        state = np.zeros(layout.dim)
        state[:3] = config.reference_xyz if self.coordinates_held else approximate_xyz
        variances = np.full(layout.dim, 1e6)
        variances[:3] = 1e-8 if self.coordinates_held else 1e4
        variances[layout.all_clocks] = 1e10
        if layout.tropo is not None:
            state[layout.tropo] = self.atmosphere.initial_zwd
            variances[layout.tropo] = self.atmosphere.initial_zwd_variance
        self.kf = ExtendedKalmanFilter(state, np.diag(variances))
        self.detector = SlipDetector(gf_jump=getattr(config, 'gf_slip_m', .05) if self.options.is_ekf else None)
        self.diagnostics, self.correction = [], np.zeros(layout.dim)
        self.selector = None              # processing.selection.SatelliteSelector (float PPP)

    def satellite_terms(self, epoch, frame):
        """Coordinates, slant delays and the satellite relativistic clock term (m, per row)."""
        coords = frame[['x', 'y', 'z']].to_numpy()
        delays = self.atmosphere.slant(epoch, self.kf.x[:3], coords, frame.elevation.to_numpy(), frame.sat.to_numpy())
        relativity = (frame.relativity_m.to_numpy() if self.options.physical_corrections and 'relativity_m' in frame
                      else np.zeros(len(frame)))
        return coords, delays, relativity

    def active_states(self, sats):
        """Position, clocks, ZWD and the ambiguity states of ``sats`` (sorted). States of satellites that left
        the filter are decoupled (cross-covariances zeroed) so the restricted update equals the full one."""
        layout = self.layout
        core = list(range(layout.first))+[i for s in layout.clock_of if s != 'G' for i in layout.clock_of[s]]
        amb = [layout.amb(int(s))+j for s in sats for j in range(layout.per_satellite)]
        active = np.array(sorted(set(core+amb)), int)
        previous = getattr(self, '_active', None)
        if previous is not None:
            left = np.setdiff1d(previous, active)
            if len(left):
                P = self.kf.P
                diagonal = P[left, left].copy()
                P[left, :] = 0
                P[:, left] = 0
                P[left, left] = diagonal
        self._active = active
        return active

    def run(self, model):
        """Epoch loop; ``model`` gets the prepared frame, builds the measurement function and
        records its outputs (see the two models for the interface).

        With satellite selection and ``state='drop'`` the subset is chosen before slip detection and only the
        selected satellites enter the filter (a satellite selected again starts a new arc); with ``keep`` the
        subset is chosen after the arc bookkeeping of all visible satellites and only the update uses it."""
        kf, last = self.kf, None
        drop = self.selector is not None and getattr(self.selector.options, 'state', 'keep') == 'drop'
        # Restricted (active-state) update: float PPP only; the AR models keep the full update.
        restrict = getattr(self.config, 'active_states', True) and getattr(model, 'active_states', False)
        in_filter = set()
        for epoch, frame in self.data.groupby('epok', sort=True):
            frame = frame.copy()
            frame['elevation'] = elevation(kf.x[:3], frame[['x', 'y', 'z']].to_numpy())
            frame = frame[frame.elevation >= self.mask].sort_values('sat').reset_index(drop=True)
            if len(frame) < model.min_satellites(frame):
                self.diagnostics.append(dict(epok=epoch, status='skipped', reason='too_few_satellites', satellites=len(frame)))
                continue
            if drop:
                frame = self.selector.select(epoch, frame, kf.x[:3])
            tick = time.perf_counter()
            kf.predict(self.options.process_noise(self.layout.dim, epoch-last if last is not None else 0, self.layout.tropo))
            last = epoch
            if self.layout.all_clocks:
                kf.reset(self.layout.all_clocks, 0.0, 1e10)            # receiver clocks: white noise
            slipped = list(self.detector.check(epoch, frame))
            if drop:                                                     # back in the filter: new arc
                slipped += [s for s in frame.sat.to_numpy(int) if s not in in_filter and s not in slipped]
                in_filter = set(frame.sat.to_numpy(int))
            for sat in slipped:
                model.reset_arc(self, int(sat), epoch, frame)
            model.prepare(self, epoch, frame, slipped)
            active = self.active_states(frame.sat.to_numpy(int)) if restrict else None
            core_ms = (time.perf_counter()-tick)*1e3
            if self.selector is not None and not drop:
                # Slip detection and arc bookkeeping above see every visible satellite; only the update
                # uses the selected subset (the ambiguities of excluded satellites are kept, unobserved).
                frame = self.selector.select(epoch, frame, kf.x[:3])
                if len(frame) < model.min_satellites(frame):
                    self.diagnostics.append(dict(epok=epoch, status='skipped', reason='too_few_selected', satellites=len(frame)))
                    continue
            tick = time.perf_counter()
            function, groups, classes = model.measurements(self, epoch, frame)
            prior = kf.x.copy()
            # Fading of the position covers dynamic-model errors, so only a kinematic receiver gets it; in
            # static mode it let the solution wander by decimetres (MADR 2023-11-23), while the per-class
            # variance factors (adaptive EKF) still apply.
            fading = [0, 1, 2] if self.options.dynamics == 'kinematic' else None
            report = kf.update(function, groups, robust=self.quality.robust if self.options.robust else False,
                               adaptive=self.options.adaptive, adaptive_indices=fading, classes=classes, active=active,
                               quality=self.quality, min_groups=model.min_satellites(frame))
            self.correction = kf.x-prior
            if self.selector is not None and self.selector.epochs and self.selector.epochs[-1]['epok'] == epoch:
                # Run time of the filter itself (prediction, slips, models; measurement update), without the
                # analysis bookkeeping: the computational side of satellite selection.
                self.selector.epochs[-1].update(core_ms=core_ms, update_ms=(time.perf_counter()-tick)*1e3,
                                                filter_states=len(active) if active is not None else self.layout.dim)
            model.record(self, epoch, frame, slipped, function, report)
            if self.selector is not None:
                terms = model.selection_terms(self, frame) if hasattr(model, 'selection_terms') else None
                self.selector.record_update(epoch, frame, kf.x[:3], report, terms, self.layout.all_clocks)
        return self


def receiver_clock_seconds(state, layout):
    return state[layout.clock_of['G'][0]]/C if layout.clock_of['G'] else np.nan
