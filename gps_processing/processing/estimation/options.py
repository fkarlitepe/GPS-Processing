"""Filter settings resolved from ProcessingConfig (or any object with the same attributes)."""
from dataclasses import dataclass
import numpy as np

ESTIMATORS = ('sequential', 'ekf', 'adaptive-ekf')


@dataclass
class FilterOptions:
    estimator: str = 'sequential'
    dynamics: str = 'static'
    position_noise: float = 10.0       # m/sqrt(s), kinematic random walk
    estimate_troposphere: bool = False
    troposphere_noise: float = 5.0     # mm/sqrt(h), zenith wet delay random walk
    physical_corrections: bool = False  # Saastamoinen hydrostatic + periodic relativity
    robust: bool = False
    code_sigma: float = 3.0            # raw undifferenced metres at zenith
    phase_sigma: float = 0.003

    @classmethod
    def from_config(cls, config):
        estimator = getattr(config, 'estimator', 'sequential')
        if estimator == 'sequential':
            # The sequential estimator is the unchanged v2.0 baseline used for comparison.
            return cls(code_sigma=getattr(config, 'code_sigma', 3.0), phase_sigma=getattr(config, 'phase_sigma', .003))
        return cls(estimator, getattr(config, 'dynamics', 'static'), getattr(config, 'position_noise', 10.0),
                   getattr(config, 'estimate_troposphere', True), getattr(config, 'troposphere_noise', 5.0),
                   getattr(config, 'physical_corrections', True), getattr(config, 'robust', True),
                   getattr(config, 'code_sigma', 3.0), getattr(config, 'phase_sigma', .003))

    @property
    def is_ekf(self):
        return self.estimator != 'sequential'

    @property
    def adaptive(self):
        return self.estimator == 'adaptive-ekf'

    def process_noise(self, dim, dt, troposphere_index=None):
        """Diagonal Q for constant ambiguities, random-walk ZWD and static/kinematic position."""
        q = np.zeros(dim)
        if not self.is_ekf or dt <= 0:
            return q
        if self.dynamics == 'kinematic':
            q[:3] = self.position_noise**2*dt
        if troposphere_index is not None:
            q[troposphere_index] = (self.troposphere_noise/1000)**2*dt/3600
        return q
