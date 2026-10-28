"""Float PPP with an iterated EKF: position, receiver clock, zenith wet delay, IF ambiguities.

State: XYZ (m), receiver clock (m, white noise), optional ZWD (m, random walk),
GPS01..32 ionosphere-free ambiguities (m, constant per arc).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from gps_processing.constants import C
from gps_processing.processing.geometry import elevation
from gps_processing.processing.cycle_slip import SlipDetector
from gps_processing.processing.atmosphere import Atmosphere
from gps_processing.processing.estimation import ExtendedKalmanFilter, FilterOptions
from gps_processing.processing.ml.stochastic import ElevationModel, IF_FACTOR


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


def solve(observations, satellites, approximate_xyz, config, variance_model=None, atmosphere=None):
    options = FilterOptions.from_config(config)
    if not options.is_ekf:
        options.estimator = 'ekf'
    mask = getattr(config, 'elevation_mask', 10)
    columns = ['epok', 'sat', 'x', 'y', 'z']+(['relativity_m'] if 'relativity_m' in satellites else [])
    data = observations.merge(satellites[columns], on=['epok', 'sat'], validate='one_to_one')
    weights = variance_model or ElevationModel(options.code_sigma, options.phase_sigma)
    atmosphere = atmosphere or Atmosphere(approximate_xyz, physical=options.physical_corrections)
    tropo = 4 if options.estimate_troposphere else None
    first = 5 if tropo is not None else 4
    dim = first+32
    state = np.zeros(dim); state[:3] = approximate_xyz
    variances = np.r_[[100.0**2]*3, 1e10, [atmosphere.initial_zwd_variance] if tropo is not None else [], [1e6]*32]
    if tropo is not None:
        state[tropo] = atmosphere.initial_zwd
    kf = ExtendedKalmanFilter(state, np.diag(variances))
    detector = SlipDetector(gf_jump=getattr(config, 'gf_slip_m', 0.05))
    positions, ambiguities, diagnostics, residuals = [], [], [], []
    last_epoch, correction = None, np.zeros(dim)
    for epoch, frame in data.groupby('epok', sort=True):
        frame = frame.copy()
        frame['elevation'] = elevation(kf.x[:3], frame[['x', 'y', 'z']].to_numpy())
        frame = frame[frame.elevation >= mask].reset_index(drop=True)
        if len(frame) < (5 if tropo is not None else 4):
            diagnostics.append(dict(epok=epoch, satellites=len(frame), status='yetersiz uydu'))
            continue
        kf.predict(options.process_noise(dim, epoch-last_epoch if last_epoch is not None else 0, tropo))
        last_epoch = epoch
        kf.reset(3, 0.0, 1e10)
        slipped = detector.check(epoch, frame)
        for row in frame[frame.sat.isin(slipped)].itertuples():
            kf.reset(first+int(row.sat)-1, row.Lc-row.Pc, 1e6)
        ids = frame.sat.to_numpy(dtype=int)
        frame['arc_epochs'] = [detector.arc[s] for s in ids]
        n = len(frame)
        coords = frame[['x', 'y', 'z']].to_numpy()
        satellite_clock = C*frame.dt_sat.to_numpy()*1e-6
        if options.physical_corrections and 'relativity_m' in frame:
            satellite_clock += frame.relativity_m.to_numpy()
        delays = atmosphere.slant(epoch, kf.x[:3], coords, frame.elevation.to_numpy())
        wet = delays.wet
        code_sigma, phase_sigma = weights.sigmas(frame)
        base_noise = np.r_[(IF_FACTOR*code_sigma)**2, (IF_FACTOR*phase_sigma)**2]
        measured = np.r_[frame.Pc.to_numpy(), frame.Lc.to_numpy()]
        amb = first+ids-1
        groups = np.r_[np.arange(n), np.arange(n)]

        def model(current, inflation):
            delta = current[:3]-coords
            distance = np.linalg.norm(delta, axis=1)
            h = np.zeros((2*n, dim))
            h[:n, :3] = delta/distance[:, None]
            h[n:, :3] = h[:n, :3]
            h[:, 3] = 1
            h[n+np.arange(n), amb] = 1
            code = distance+current[3]-satellite_clock+delays.common
            if tropo is not None:
                h[:, tropo] = np.r_[wet, wet]
                code = code+wet*current[tropo]
            return measured, np.r_[code+delays.code, code+delays.phase+current[amb]], h, np.diag(base_noise*inflation[groups]), groups

        prior = kf.x.copy()
        dynamic = list(range(3))
        report = kf.update(model, n, robust=options.robust, adaptive=options.adaptive, adaptive_indices=dynamic)
        correction = kf.x-prior
        _, predicted, *_ = model(kf.x, np.ones(n))
        residual = measured-predicted
        code_rms = float(np.sqrt(np.mean(residual[:n]**2)))
        phase_rms = float(np.sqrt(np.mean(residual[n:]**2)))
        sigma = np.sqrt(np.diag(kf.P)[:3])
        positions.append(dict(epok=epoch, x=kf.x[0], y=kf.x[1], z=kf.x[2], receiver_clock_s=kf.x[3]/C,
                              zwd_m=delays.zwd_apriori+kf.x[tropo] if tropo is not None else (delays.zwd_apriori if atmosphere.tro is not None else np.nan),
                              stec_tecu=float(np.nanmean(delays.stec)) if np.isfinite(delays.stec).any() else np.nan,
                              iono2_phase_mm=float(np.mean(np.abs(delays.phase)))*1e3,
                              sigma_x=sigma[0], sigma_y=sigma[1], sigma_z=sigma[2], satellites=n,
                              code_rms_m=code_rms, phase_rms_m=phase_rms, nis=report.nis, alpha=report.alpha,
                              downweighted=report.downweighted, status='float'))
        ambiguities.extend(dict(epok=epoch, sat=int(s), N3=kf.x[first+s-1]) for s in ids)
        diagnostics.append(dict(epok=epoch, satellites=n, status='çözüldü', code_rms_m=code_rms,
                                reset_satellites=','.join(map(str, slipped)), nis=report.nis,
                                alpha=report.alpha, downweighted=report.downweighted))
        residuals.append(pd.DataFrame(dict(epok=epoch, sat=ids, elevation=frame.elevation.to_numpy(),
                                           snr1=frame.get('snr1', np.nan), snr2=frame.get('snr2', np.nan),
                                           arc_epochs=frame.arc_epochs.to_numpy(), code_res=residual[:n],
                                           phase_res=residual[n:], inflation=report.inflation)))
    if not positions:
        raise ValueError('Çözülebilen epok yok: yükseklik maskesini ve gözlem verilerini kontrol edin.')
    return EstimationOutput(pd.DataFrame(positions), pd.DataFrame(ambiguities), pd.DataFrame(diagnostics),
                            kf.P, correction, pd.concat(residuals, ignore_index=True))
