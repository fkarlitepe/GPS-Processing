"""Iterated extended Kalman filter shared by float PPP and the PPP-AR user models.

Measurement models are callables ``model(state, inflation)`` returning
``(measured, predicted, design, noise, groups)``; ``groups`` maps each
measurement row to a satellite so robust down-weighting inflates the whole
satellite noise block (needed when rows are correlated, e.g. SDBS).

Adaptive EKF: besides the innovation-based fading of the dynamic states, every observation
class (code / phase / MW of each system, ``classes``) gets its own variance factor estimated from
the predicted innovations (Mehra 1972 / Sage-Husa form, smoothed over epochs). The receiver
clock (white noise, common to all rows of a class) would dominate the innovation variance, so the
innovations are centred over the satellites of the class, which removes it exactly:
    factor_g = sum (vc_i^2 - (Hc P Hc^T)_ii) / sum (C R0 C^T)_ii,   C = I - 11^T/m
using only rows the prior predicts well (new ambiguities are left out). This gives each
observation group its own adaptive factor (the "multiple adaptive factors" left as future work
by Guo & Zhang 2014). Post-fit residuals are not used: the estimate would feed back on itself.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2


@dataclass
class UpdateReport:
    nis: float = np.nan          # normalized innovation squared per measurement, E = 1
    alpha: float = 1.0           # adaptive inflation applied to the predicted dynamic states
    inflation: np.ndarray = field(default_factory=lambda: np.ones(0))
    iterations: int = 0
    innovation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    groups: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    factors: dict = field(default_factory=dict)   # adaptive variance factor per observation class

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
    def __init__(self, state, covariance, smoothing=0.05, factor_limits=(0.2, 25.0)):
        self.x = np.asarray(state, dtype=float).copy()
        self.P = np.asarray(covariance, dtype=float).copy()
        self.smoothing, self.factor_limits = smoothing, factor_limits
        self.components = {}                 # class -> [weighted sum of e^2/R, weighted redundancy]

    def factor(self, group):
        numerator, cofactor = self.components.get(group, (0.0, 0.0))
        if cofactor <= 0:
            return 1.0
        return float(np.clip(numerator/cofactor, *self.factor_limits))

    def _learn_components(self, classes, innovation, h, prior_cov, variance, cofactor):
        keep = 1-self.smoothing
        for group in np.unique(classes):
            rows = np.flatnonzero(classes == group)
            hg = h[rows]
            common = np.all(hg == hg[:1], axis=0)                 # e.g. the receiver clock of the class
            local = np.where(common[None, :], 0.0, hg)
            ok = rows[np.einsum('ij,jk,ik->i', local, prior_cov, local) < variance[rows]]
            m = len(ok)
            if m < 3:
                continue
            centred_h = h[ok]-h[ok].mean(axis=0)
            centred_v = innovation[ok]-innovation[ok].mean()
            predicted = np.einsum('ij,jk,ik->i', centred_h, prior_cov, centred_h)
            r0 = cofactor[ok]
            centred_r = r0*(1-2/m)+r0.sum()/m**2
            old = self.components.get(int(group), (0.0, 0.0))
            self.components[int(group)] = (keep*old[0]+float(np.sum(centred_v**2-predicted)), keep*old[1]+float(centred_r.sum()))

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

    def constrain(self, indices, values, variance):
        """Linear update with direct pseudo-observations x[indices] = values (fix-and-hold)."""
        indices = np.asarray(indices, int)
        h = np.zeros((len(indices), len(self.x)))
        h[np.arange(len(indices)), indices] = 1
        s = h@self.P@h.T+np.eye(len(indices))*variance
        gain = np.linalg.solve(s, h@self.P).T
        self.x = self.x+gain@(np.asarray(values, float)-self.x[indices])
        identity = np.eye(len(self.x))-gain@h
        cov = identity@self.P@identity.T+gain@(np.eye(len(indices))*variance)@gain.T
        self.P = (cov+cov.T)/2

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
               robust=False, adaptive=False, adaptive_indices=None, adaptive_threshold=None, classes=None):
        """Gauss-Newton iterated update in Joseph form; optional IGG-III, fading and (adaptive with
        ``classes``) per-class variance factors."""
        factors = {}
        if adaptive and classes is not None:
            classes = np.asarray(classes)
            factors = {int(g): self.factor(int(g)) for g in np.unique(classes)}
            scale = np.sqrt(np.array([factors[int(g)] for g in classes]))
            raw = model

            def model(state, inflation):
                measured, predicted, h, r, groups = raw(state, inflation)
                return measured, predicted, h, r*scale[:, None]*scale[None, :], groups
        prior, prior_cov = self.x.copy(), self.P.copy()
        inflation = np.ones(groups_count)
        if factors:
            measured, predicted, h, r, _ = model(prior, inflation)
            self._learn_components(classes, measured-predicted, h, prior_cov, np.diag(r), np.diag(r)/scale**2)
        alpha = 1.0
        if adaptive and adaptive_indices is not None:
            measured, predicted, h, r, _ = model(prior, inflation)
            v = measured-predicted
            statistic = float(v@np.linalg.solve(h@prior_cov@h.T+r, v)/len(v))
            # Innovation-based fading factor: inflate only position/troposphere when the
            # innovation is inconsistent with the predicted covariance.
            # Fading only on a significant mismatch: the chi-square 99.9 % quantile of the NIS per
            # measurement (a fixed 1.5 inflated the position whenever the noise was mis-scaled and
            # let a static solution drift by decimetres; the class factors absorb the mis-scaling).
            threshold = adaptive_threshold or chi2.ppf(0.999, len(v))/len(v)
            if statistic > threshold:
                alpha = statistic/threshold
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
        return UpdateReport(nis, alpha, inflation, count, innovation, groups, factors)
