"""Iterated extended Kalman filter shared by float PPP and the PPP-AR user models.

Measurement models are callables ``model(state, inflation)`` returning
``(measured, predicted, design, noise, groups)``; ``groups`` maps each
measurement row to a satellite so robust down-weighting inflates the whole
satellite noise block (needed when rows are correlated, e.g. SDBS).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class UpdateReport:
    nis: float = np.nan          # normalized innovation squared per measurement, E = 1
    alpha: float = 1.0           # adaptive inflation applied to the predicted dynamic states
    inflation: np.ndarray = field(default_factory=lambda: np.ones(0))
    iterations: int = 0
    innovation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    groups: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def downweighted(self):
        return int(np.sum(self.inflation > 1))


def igg3(standardized, k0=2.5, k1=6.0):
    """IGG-III equivalent-weight factor (Yang 1994); returns variance inflation >= 1."""
    v = np.abs(standardized)
    weight = np.ones_like(v)
    mid = (v > k0) & (v <= k1)
    weight[mid] = k0/v[mid]*((k1-v[mid])/(k1-k0))**2
    weight[v > k1] = 1e-8
    return 1/np.maximum(weight, 1e-8)


class ExtendedKalmanFilter:
    def __init__(self, state, covariance):
        self.x = np.asarray(state, dtype=float).copy()
        self.P = np.asarray(covariance, dtype=float).copy()

    def predict(self, noise=None, transition=None):
        if transition is not None:
            self.x = transition@self.x
            self.P = transition@self.P@transition.T
        if noise is not None:
            self.P = self.P+(np.diag(noise) if np.ndim(noise) == 1 else noise)

    def reset(self, indices, value=0.0, variance=1e6):
        indices = np.atleast_1d(indices)
        self.x[indices] = value
        self.P[indices, :] = 0
        self.P[:, indices] = 0
        self.P[indices, indices] = variance

    def _iterate(self, model, prior, prior_cov, inflation, iterations, tolerance):
        state = prior.copy()
        for count in range(1, iterations+1):
            measured, predicted, h, r, groups = model(state, inflation)
            innovation = measured-predicted+h@(state-prior)
            s = h@prior_cov@h.T+r
            gain = np.linalg.solve(s, h@prior_cov).T
            updated = prior+gain@innovation
            change = np.linalg.norm(updated[:3]-state[:3])
            state = updated
            if change < tolerance:
                break
        return state, gain, h, r, s, innovation, groups, count

    def update(self, model, groups_count, iterations=8, tolerance=1e-4,
               robust=False, adaptive=False, adaptive_indices=None, adaptive_threshold=1.5):
        """Gauss-Newton iterated update in Joseph form; optional IGG-III and fading."""
        prior, prior_cov = self.x.copy(), self.P.copy()
        inflation = np.ones(groups_count)
        alpha = 1.0
        if adaptive and adaptive_indices is not None:
            measured, predicted, h, r, _ = model(prior, inflation)
            v = measured-predicted
            statistic = float(v@np.linalg.solve(h@prior_cov@h.T+r, v)/len(v))
            # Innovation-based fading factor: inflate only position/troposphere when the
            # innovation is inconsistent with the predicted covariance.
            if statistic > adaptive_threshold:
                alpha = statistic/adaptive_threshold
                idx = np.asarray(adaptive_indices)
                prior_cov[np.ix_(idx, idx)] *= alpha
        for _ in range(5 if robust else 1):
            state, gain, h, r, s, innovation, groups, count = self._iterate(
                model, prior, prior_cov, inflation, iterations, tolerance)
            if not robust:
                break
            # Post-fit residuals standardized by their own covariance R - H P+ H^T; unlike
            # predicted innovations they stay informative when the prior is still diffuse.
            measured, predicted, *_ = model(state, inflation)
            residual = measured-predicted
            posterior = prior_cov-gain@h@prior_cov
            variance = np.diag(r)-np.einsum('ij,jk,ik->i', h, posterior, h)
            standardized = residual*np.sqrt(inflation[groups]/np.maximum(variance, 1e-12))
            worst = np.zeros(groups_count)
            np.maximum.at(worst, groups, np.abs(standardized))
            updated = np.maximum(inflation, igg3(worst))
            if np.allclose(updated, inflation):
                break
            inflation = updated
        identity = np.eye(len(state))-gain@h
        cov = identity@prior_cov@identity.T+gain@r@gain.T
        self.x, self.P = state, (cov+cov.T)/2
        nis = float(innovation@np.linalg.solve(s, innovation)/len(innovation))
        return UpdateReport(nis, alpha, inflation, count, innovation, groups)
