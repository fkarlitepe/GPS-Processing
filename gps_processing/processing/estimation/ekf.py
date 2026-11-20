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
from .quality import (QualityOptions, TestResult, EXCLUDED, DIFFUSE, evaluate, identify, igg3,
                      robust_inflation)


@dataclass
class UpdateReport:
    nis: float = np.nan          # normalized innovation squared per measurement, E = 1
    alpha: float = 1.0           # adaptive inflation applied to the predicted dynamic states
    inflation: np.ndarray = field(default_factory=lambda: np.ones(0))
    iterations: int = 0
    innovation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    groups: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    factors: dict = field(default_factory=dict)   # adaptive variance factor per observation class
    gain: np.ndarray = None      # Kalman gain of the final iteration (x+ = x- + gain @ innovation)
    design: np.ndarray = None    # design matrix H of the final iteration
    test: TestResult = None      # outlier test (global / Baarda / Pope) of the update, if requested

    @property
    def downweighted(self):
        return int(np.sum(self.inflation > 1))


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
               robust=False, adaptive=False, adaptive_indices=None, adaptive_threshold=None, classes=None, active=None,
               quality=None, min_groups=1):
        """Gauss-Newton iterated update in Joseph form; optional IGG-III, fading and (adaptive with
        ``classes``) per-class variance factors.

        ``robust``: False, True ('igg3') or the weight function 'igg3' / 'huber' / 'danish' (``quality`` holds the
        constants). ``quality`` (QualityOptions): outlier test 'global' / 'baarda' / 'pope' of the update
        (processing.estimation.quality); identified groups are excluded one at a time, keeping ``min_groups``.

        ``active`` (sorted state indices starting with the position): the update runs on this sub-state only;
        the other states must be uncorrelated with it (their rows/columns of P are zero off the diagonal), so the
        result equals the full update at the cost of the active dimension. Gain and design of the report are
        returned in full size."""
        if active is not None and len(active) < len(self.x):
            a = np.asarray(active, int)
            full_x, full_p = self.x, self.P
            frozen = full_x.copy()

            def sub_model(state, inflation, _raw=model):
                current = frozen.copy()
                current[a] = state
                measured, predicted, h, r, groups = _raw(current, inflation)
                return measured, predicted, h[:, a], r, groups
            self.x, self.P = full_x[a].copy(), full_p[np.ix_(a, a)].copy()
            try:
                report = self.update(sub_model, groups_count, iterations, tolerance, robust, adaptive,
                                     adaptive_indices, adaptive_threshold, classes, None, quality, min_groups)
            finally:
                full_x[a], full_p[np.ix_(a, a)] = self.x, self.P
                self.x, self.P = full_x, full_p
            gain = np.zeros((len(full_x), report.gain.shape[1]))
            gain[a] = report.gain
            design = np.zeros((report.design.shape[0], len(full_x)))
            design[:, a] = report.design
            report.gain, report.design = gain, design
            return report
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
        method = ('igg3' if robust is True else robust) if robust else None
        quality = quality or QualityOptions()
        test, done = None, None
        if quality.active:
            test, done = self._outlier_test(model, prior, prior_cov, inflation, iterations, tolerance, quality, min_groups)
        for loop in range(5 if method else 1):
            if done is not None and loop == 0:
                state, gain, h, r, s, innovation, groups, count = done
            else:
                state, gain, h, r, s, innovation, groups, count = self._iterate(
                    model, prior, prior_cov, inflation, iterations, tolerance)
            if not method:
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
            updated = np.maximum(inflation, robust_inflation(method, worst, quality))
            if test is not None:
                updated[test.excluded] = EXCLUDED
            if np.allclose(updated, inflation):
                break
            inflation = updated
        identity = np.eye(len(state))-gain@h
        cov = identity@prior_cov@identity.T+gain@r@gain.T
        self.x, self.P = state, (cov+cov.T)/2
        nis = float(innovation@np.linalg.solve(s, innovation)/len(innovation))
        return UpdateReport(nis, alpha, inflation, count, innovation, groups, factors, gain, h, test)

    def _outlier_test(self, model, prior, prior_cov, inflation, iterations, tolerance, quality, min_groups):
        """DIA on the update: global test, then Baarda w / Pope tau of the post-fit residuals; the group of the
        largest significant statistic gets the EXCLUDED inflation and the update is repeated. ``inflation`` is
        changed in place; returns (TestResult, last _iterate output)."""
        result, first = TestResult(quality.test), None
        while True:
            out = self._iterate(model, prior, prior_cov, inflation, iterations, tolerance)
            state, gain, h, r, s, innovation, groups, count = out
            residual = innovation-h@(gain@innovation)                 # linearized post-fit residual = R S^-1 e
            posterior = prior_cov-gain@h@prior_cov
            residual_cov = r-h@posterior@h.T
            weight = np.linalg.inv(r)
            kept = ~np.isin(groups, result.excluded)
            rows = np.flatnonzero(kept)
            statistic = max(float(innovation[rows]@(weight@residual)[rows]), 0.0)
            observed = np.any(h[rows] != 0, axis=0)
            diffuse = int(np.sum(observed & (np.diag(prior_cov) >= DIFFUSE)))
            df = len(rows)-diffuse
            result.statistic, result.df = statistic, df
            result.sigma0 = float(np.sqrt(statistic/df)) if df > 0 else np.nan
            if df < 1:
                break
            crit, rejected, local, local_crit = evaluate(quality, residual, weight, residual_cov, statistic, df, rows)
            result.critical, result.passed, result.local_critical = crit, not rejected, local_crit
            result.local_max = float(np.max(np.abs(local[rows]))) if quality.test != 'global' and len(rows) else np.nan
            if first is None:
                first = result.first_passed = not rejected
            if not rejected or quality.test == 'global' or len(result.excluded) >= quality.max_exclusions:
                break
            group = identify(local, local_crit, groups, result.excluded, min_groups, len(inflation))
            if group is None:
                break
            result.excluded.append(group)
            inflation[group] = EXCLUDED
        return result, out
