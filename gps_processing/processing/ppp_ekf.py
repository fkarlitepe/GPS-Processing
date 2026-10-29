"""Float PPP with the shared iterated EKF (processing.engine): position, receiver clock, zenith wet
delay, IF ambiguities.

State: XYZ (m), receiver clock (m, white noise), optional ZWD (m, random walk), ionosphere-free
ambiguities (m, constant per arc) for GPS G01..G32 and Galileo E01..E36 (gnss.slot), and a
Galileo receiver clock (m, white noise; the GPS-Galileo system time offset and inter-system
code bias are absorbed in it). GPS uses L1/L2, Galileo E1/E5a.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from gps_processing.constants import C
from gps_processing.processing.engine import FilterEngine, Layout
from gps_processing.processing.estimation import FilterOptions
from gps_processing.processing.ml.stochastic import ElevationModel
from gps_processing import gnss


@dataclass
class EstimationOutput:
    positions: pd.DataFrame
    ambiguities: pd.DataFrame
    diagnostics: pd.DataFrame
    covariance: np.ndarray
    correction: np.ndarray
    residuals: pd.DataFrame

    def __iter__(self):
        return iter((self.positions, self.ambiguities, self.diagnostics, self.covariance, self.correction))


class IonosphereFreeModel:
    """Observation plug-in: IF code and phase rows, one IF ambiguity (m) per satellite."""

    def __init__(self, weights, layout):
        self.weights, self.layout = weights, layout
        self.arc_id = {}                   # increments at every arc restart (used by FCB estimation)
        self.positions, self.ambiguities, self.residuals = [], [], []

    def min_satellites(self, frame):
        # One more unknown (Galileo clock) when both systems are tracked.
        both = (frame.sat > 100).any() and (frame.sat <= 100).any()
        return (5 if self.layout.tropo is not None else 4)+int(both)

    def reset_arc(self, engine, sat, epoch, frame):
        row = frame[frame.sat == sat].iloc[0]
        engine.kf.reset(self.layout.amb(sat), row.Lc-row.Pc, 1e6)
        self.arc_id[sat] = self.arc_id.get(sat, -1)+1

    def prepare(self, engine, epoch, frame, slipped):
        frame['arc_epochs'] = [engine.detector.arc[s] for s in frame.sat.to_numpy(int)]

    def measurements(self, engine, epoch, frame):
        layout, ids = self.layout, frame.sat.to_numpy(dtype=int)
        n = len(frame)
        coords, delays, relativity = engine.satellite_terms(epoch, frame)
        satellite_clock = C*frame.dt_sat.to_numpy()*1e-6+relativity
        wet = delays.wet
        code_sigma, phase_sigma = self.weights.sigmas(frame)
        factor = gnss.per_row(ids, 'if_factor')
        base_noise = np.r_[(factor*code_sigma)**2, (factor*phase_sigma)**2]
        measured = np.r_[frame.Pc.to_numpy(), frame.Lc.to_numpy()]
        amb = np.array([layout.amb(s) for s in ids], int)
        clock_column = np.where(ids > 100, layout.clock_of['E'][0], layout.clock_of['G'][0])
        groups = np.r_[np.arange(n), np.arange(n)]
        tropo = layout.tropo
        dim = layout.dim

        def model(current, inflation):
            delta = current[:3]-coords
            distance = np.linalg.norm(delta, axis=1)
            h = np.zeros((2*n, dim))
            h[:n, :3] = delta/distance[:, None]
            h[n:, :3] = h[:n, :3]
            h[np.arange(2*n), np.r_[clock_column, clock_column]] = 1
            h[n+np.arange(n), amb] = 1
            code = distance+current[clock_column]-satellite_clock+delays.common
            if tropo is not None:
                h[:, tropo] = np.r_[wet, wet]
                code = code+wet*current[tropo]
            return measured, np.r_[code+delays.code, code+delays.phase+current[amb]], h, np.diag(base_noise*inflation[groups]), groups

        self.delays = delays
        # Variance classes (adaptive EKF): code / phase per system.
        system = (ids > 100).astype(int)*2
        return model, n, np.r_[system, system+1]

    def record(self, engine, epoch, frame, slipped, model, report):
        kf, layout, atmosphere, delays = engine.kf, self.layout, engine.atmosphere, self.delays
        ids, n = frame.sat.to_numpy(dtype=int), len(frame)
        tropo = layout.tropo
        measured, predicted, *_ = model(kf.x, np.ones(n))
        residual = measured-predicted
        code_rms = float(np.sqrt(np.mean(residual[:n]**2)))
        phase_rms = float(np.sqrt(np.mean(residual[n:]**2)))
        sigma = np.sqrt(np.diag(kf.P)[:3])
        self.positions.append(dict(epok=epoch, x=kf.x[0], y=kf.x[1], z=kf.x[2], receiver_clock_s=kf.x[layout.clock_of['G'][0]]/C,
                                   zwd_m=delays.zwd_apriori+kf.x[tropo] if tropo is not None else (delays.zwd_apriori if atmosphere.tro is not None else np.nan),
                                   stec_tecu=float(np.nanmean(delays.stec)) if np.isfinite(delays.stec).any() else np.nan,
                                   iono2_phase_mm=float(np.mean(np.abs(delays.phase)))*1e3,
                                   sigma_x=sigma[0], sigma_y=sigma[1], sigma_z=sigma[2], satellites=n,
                                   code_rms_m=code_rms, phase_rms_m=phase_rms, nis=report.nis, alpha=report.alpha,
                                   downweighted=report.downweighted, status='float',
                                   gps_satellites=int((ids <= 100).sum()), galileo_satellites=int((ids > 100).sum()),
                                   **{f'variance_factor_{k}': v for k, v in report.factors.items()}))
        self.ambiguities.extend(dict(epok=epoch, sat=int(s), N3=kf.x[layout.amb(s)], sigma_N3=float(np.sqrt(kf.P[layout.amb(s), layout.amb(s)])),
                                     arc=self.arc_id.get(int(s), 0), elevation=float(e)) for s, e in zip(ids, frame.elevation))
        engine.diagnostics.append(dict(epok=epoch, satellites=n, status='çözüldü', code_rms_m=code_rms,
                                       reset_satellites=','.join(map(str, slipped)), nis=report.nis,
                                       alpha=report.alpha, downweighted=report.downweighted))
        self.residuals.append(pd.DataFrame(dict(epok=epoch, sat=ids, elevation=frame.elevation.to_numpy(),
                                                snr1=frame.get('snr1', np.nan), snr2=frame.get('snr2', np.nan),
                                                arc_epochs=frame.arc_epochs.to_numpy(), code_res=residual[:n],
                                                phase_res=residual[n:], inflation=report.inflation)))


def solve(observations, satellites, approximate_xyz, config, variance_model=None, atmosphere=None):
    options = FilterOptions.from_config(config)
    if not options.is_ekf:
        options.estimator = 'ekf'
    layout = Layout(clocks=1, troposphere=options.estimate_troposphere, per_satellite=1)
    weights = variance_model or ElevationModel(options.code_sigma, options.phase_sigma)
    engine = FilterEngine(observations, satellites, approximate_xyz, config, layout, atmosphere, options)
    model = IonosphereFreeModel(weights, layout)
    engine.run(model)
    if not model.positions:
        raise ValueError('Çözülebilen epok yok: yükseklik maskesini ve gözlem verilerini kontrol edin.')
    return EstimationOutput(pd.DataFrame(model.positions), pd.DataFrame(model.ambiguities), pd.DataFrame(engine.diagnostics),
                            engine.kf.P, engine.correction, pd.concat(model.residuals, ignore_index=True))
