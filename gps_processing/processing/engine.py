"""One EKF core for float PPP and the PPP-AR user models.

The epoch loop, state layout, elevation mask, prediction, receiver-clock and cycle-slip resets,
atmospheric delays, satellite clock terms and the robust/adaptive update live here once;
an observation model (plug-in) supplies the measurement equations and its own bookkeeping:

* ``processing.ppp_ekf.IonosphereFreeModel``: IF code + phase, one IF ambiguity per satellite;
* ``processing.ar.solver.UserModel``: IF code, IF phase and MW with (N1, NW) per satellite and the
  SDBS / DCM / IPC transformation; integer fixing runs after the float update (fixed
  estimates never feed the filter unless fix-and-hold is selected).

State layout of both:
    [X Y Z | receiver clocks of the first system (0-3) | ZWD | ambiguities (gnss.SLOTS x k) |
     Galileo receiver clocks (same number)]
"""
import numpy as np
from gps_processing.constants import C
from gps_processing.processing.geometry import elevation
from gps_processing.processing.cycle_slip import SlipDetector
from gps_processing.processing.atmosphere import Atmosphere
from gps_processing.processing.estimation import ExtendedKalmanFilter, FilterOptions
from gps_processing import gnss


class Layout:
    def __init__(self, clocks, troposphere, per_satellite):
        self.clocks, self.per_satellite = clocks, per_satellite
        base = 3+clocks
        self.tropo = base if troposphere else None
        self.first = base+bool(troposphere)
        self.galileo = self.first+per_satellite*gnss.SLOTS
        self.dim = self.galileo+clocks
        self.clock_of = {'G': list(range(3, base)), 'E': list(range(self.galileo, self.dim))}
        self.all_clocks = self.clock_of['G']+self.clock_of['E']

    def amb(self, sat):
        """First ambiguity state of ``sat`` (AR: N1, NW follows)."""
        return self.first+self.per_satellite*gnss.slot(sat)


class FilterEngine:
    def __init__(self, observations, satellites, approximate_xyz, config, layout, atmosphere=None, options=None):
        self.config, self.layout = config, layout
        self.options = options or FilterOptions.from_config(config)
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

    def satellite_terms(self, epoch, frame):
        """Coordinates, slant delays and the satellite relativistic clock term (m, per row)."""
        coords = frame[['x', 'y', 'z']].to_numpy()
        delays = self.atmosphere.slant(epoch, self.kf.x[:3], coords, frame.elevation.to_numpy(), frame.sat.to_numpy())
        relativity = (frame.relativity_m.to_numpy() if self.options.physical_corrections and 'relativity_m' in frame
                      else np.zeros(len(frame)))
        return coords, delays, relativity

    def run(self, model):
        """Epoch loop; ``model`` gets the prepared frame, builds the measurement function and
        records its outputs (see the two models for the interface)."""
        kf, last = self.kf, None
        for epoch, frame in self.data.groupby('epok', sort=True):
            frame = frame.copy()
            frame['elevation'] = elevation(kf.x[:3], frame[['x', 'y', 'z']].to_numpy())
            frame = frame[frame.elevation >= self.mask].sort_values('sat').reset_index(drop=True)
            if len(frame) < model.min_satellites(frame):
                self.diagnostics.append(dict(epok=epoch, status='skipped', reason='too_few_satellites', satellites=len(frame)))
                continue
            kf.predict(self.options.process_noise(self.layout.dim, epoch-last if last is not None else 0, self.layout.tropo))
            last = epoch
            if self.layout.all_clocks:
                kf.reset(self.layout.all_clocks, 0.0, 1e10)            # receiver clocks: white noise
            slipped = self.detector.check(epoch, frame)
            for sat in slipped:
                model.reset_arc(self, int(sat), epoch, frame)
            model.prepare(self, epoch, frame, slipped)
            function, groups, classes = model.measurements(self, epoch, frame)
            prior = kf.x.copy()
            # Fading of the position covers dynamic-model errors, so only a kinematic receiver gets it; in
            # static mode it let the solution wander by decimetres (MADR 2023-11-23), while the per-class
            # variance factors (adaptive EKF) still apply.
            fading = [0, 1, 2] if self.options.dynamics == 'kinematic' else None
            report = kf.update(function, groups, robust=self.options.robust, adaptive=self.options.adaptive,
                               adaptive_indices=fading, classes=classes)
            self.correction = kf.x-prior
            model.record(self, epoch, frame, slipped, function, report)
        return self


def receiver_clock_seconds(state, layout):
    return state[layout.clock_of['G'][0]]/C if layout.clock_of['G'] else np.nan
