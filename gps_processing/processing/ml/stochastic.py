"""Observation weighting: elevation model and a residual-trained (ML) stochastic model.

All ``sigmas`` methods return raw undifferenced code/phase standard deviations (metres),
so the float PPP filter scales them to the ionosphere-free combination and the AR
filter propagates them through ``combination_covariance``.
"""
import numpy as np
from gps_processing.constants import F1, F2
from .learners import GradientBoostingRegressor

IF_FACTOR = np.sqrt(F1**4+F2**4)/(F1**2-F2**2)
LOG_CHI2_BIAS = 1.2704  # -E[log chi2(1)]: corrects log-squared residual regression to variance
FEATURES = ['sin_elevation', 'snr1', 'snr2', 'log_arc']


class ElevationModel:
    def __init__(self, code_sigma=3.0, phase_sigma=0.003):
        self.code_sigma, self.phase_sigma = code_sigma, phase_sigma

    def sigmas(self, frame):
        sine = np.maximum(np.sin(np.radians(frame.elevation.to_numpy(dtype=float))), 0.1)
        return self.code_sigma/sine, self.phase_sigma/sine


def features(frame, fill):
    sine = np.sin(np.radians(frame.elevation.to_numpy(dtype=float)))
    columns = [sine]
    for name in ('snr1', 'snr2'):
        values = frame[name].to_numpy(dtype=float) if name in frame else np.full(len(frame), np.nan)
        columns.append(np.where(np.isfinite(values), values, fill[name]))
    arc = frame['arc_epochs'].to_numpy(dtype=float) if 'arc_epochs' in frame else np.ones(len(frame))
    columns.append(np.log1p(arc))
    return np.column_stack(columns)


class LearnedVarianceModel:
    """Gradient-boosted regression of log squared post-fit residuals on elevation, SNR and
    arc length; a data-driven replacement of the fixed sin(elevation) weighting."""

    def __init__(self, fallback, code=None, phase=None, fill=None, limits=(0.3, 30.0), samples=0):
        self.fallback, self.code, self.phase = fallback, code, phase
        self.fill = fill or {'snr1': 45.0, 'snr2': 40.0}
        self.limits, self.samples = limits, samples

    @classmethod
    def fit(cls, residuals, fallback, seed=0):
        frame = residuals.replace([np.inf, -np.inf], np.nan).dropna(subset=['code_res', 'phase_res', 'elevation'])
        if len(frame) < 40:
            raise ValueError('ML ağırlık modeli için yeterli artık yok (en az 40 gözlem).')
        fill = {name: float(np.nanmedian(frame[name])) if name in frame and frame[name].notna().any() else 45.0
                for name in ('snr1', 'snr2')}
        x = features(frame, fill)
        models = []
        for column, floor in (('code_res', 0.01), ('phase_res', 1e-4)):
            # Residuals are ionosphere-free; convert to raw-signal equivalent before learning.
            target = np.log((frame[column].to_numpy(dtype=float)/IF_FACTOR)**2+floor**2)
            models.append(GradientBoostingRegressor(seed=seed).fit(x, target))
        return cls(fallback, *models, fill=fill, samples=len(frame))

    def sigmas(self, frame):
        code0, phase0 = self.fallback.sigmas(frame)
        x = features(frame, self.fill)
        low, high = self.limits
        code = np.clip(np.sqrt(np.exp(self.code.predict(x)+LOG_CHI2_BIAS)), low*code0, high*code0)
        phase = np.clip(np.sqrt(np.exp(self.phase.predict(x)+LOG_CHI2_BIAS)), low*phase0, high*phase0)
        return code, phase

    def to_dict(self):
        return {'features': FEATURES, 'fill': self.fill, 'limits': list(self.limits), 'samples': self.samples,
                'code': self.code.to_dict(), 'phase': self.phase.to_dict()}
