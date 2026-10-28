"""Checks against Ps-LAMBDA (Verhagen, Li & Teunissen 2013); equation numbers refer to the paper."""
import math
import unittest
import numpy as np
from gps_processing.processing.ar.integer import (bootstrap, chi2_cdf, decorrelate, evaluate, ldl,
                                                  simulate_success, success_rates)

# Eq. (38): dual-frequency geometry-free GPS model, one satellite-receiver pair (cycles^2).
PAPER_Q = np.array([[1.2429, 0.9683], [0.9683, 0.7547]])


class SuccessRateTests(unittest.TestCase):
    def test_chi2_cdf_closed_forms(self):
        for x in (0.1, 1.0, 3.0, 9.0):
            self.assertAlmostEqual(chi2_cdf(x, 2), 1-math.exp(-x/2), places=12)
            self.assertAlmostEqual(chi2_cdf(x, 1), math.erf(math.sqrt(x/2)), places=12)

    def test_adop_is_transformation_invariant(self):
        z, _, d = decorrelate(PAPER_Q)
        self.assertAlmostEqual(np.prod(d), np.linalg.det(PAPER_Q), places=10)           # eq. (31)
        self.assertAlmostEqual(success_rates(PAPER_Q)['adop'], np.linalg.det(PAPER_Q)**0.25, places=10)

    def test_bootstrapping_formula_is_exact(self):
        # Eq. (28) is exact for IB; check it against the IB estimator of eq. (15) itself.
        rates = success_rates(4*PAPER_Q)
        z, _, _ = decorrelate(4*PAPER_Q)
        qz = z.T@(4*PAPER_Q)@z
        draws = np.random.default_rng(1).multivariate_normal(np.zeros(2), qz, 20000)
        empirical = np.mean([not bootstrap(x, qz).any() for x in draws])
        self.assertAlmostEqual(empirical, rates['ib'], delta=4*math.sqrt(rates['ib']*(1-rates['ib'])/20000))

    def test_decorrelation_matters_for_ib(self):
        # Sec. 3.3 / Table 2: IB on the original ambiguities is far worse than after Z-decorrelation.
        rates = success_rates(PAPER_Q)
        self.assertLess(rates['ib_original'], 0.5)
        self.assertGreater(rates['ib'], 0.99)

    def test_bounds_bracket_simulation_and_orderings(self):
        for scale in (1.0, 2.0, 5.0):
            q = scale*PAPER_Q
            r = success_rates(q)
            ils = simulate_success(q, 'ils', 8000, seed=2)
            noise = 4*math.sqrt(max(ils*(1-ils), 1e-4)/8000)
            self.assertLessEqual(r['ir_lower'], r['ib']+1e-12)                   # eqs. (24), (26)
            self.assertLessEqual(r['ib'], ils+noise)                             # eq. (32)
            self.assertLessEqual(ils, r['ils_upper']+noise)                      # eq. (33)
            self.assertLessEqual(r['ib'], r['adop_approx']+1e-12)                # eq. (30)
            self.assertLessEqual(r['eig_lower'], r['ib']+1e-12)                  # eq. (37)
            self.assertGreaterEqual(r['eig_upper'], r['ils_upper']-1e-12)

    def test_evaluate_reports_decorrelated_success_for_ils(self):
        a = np.array([0.2, -0.1])
        ils = evaluate(a, PAPER_Q, 'ils', 0.0)
        lam = evaluate(a, PAPER_Q, 'lambda', 0.0)
        self.assertAlmostEqual(ils['success'], success_rates(PAPER_Q)['ib'], places=12)
        self.assertAlmostEqual(lam['success'], ils['success'], places=12)
        self.assertGreaterEqual(ils['success_upper'], ils['success'])
        np.testing.assert_equal(ils['combinations'], np.eye(2))
        # ILS is Z-invariant: the LAMBDA full-set solution maps to the same integers.
        np.testing.assert_allclose(np.linalg.solve(lam['combinations'], lam['integers']), ils['integers'])
        self.assertAlmostEqual(ils['ratio'], lam['ratio'], places=9)

    def test_ldl_lambda_order(self):
        lower, d = ldl(PAPER_Q)
        np.testing.assert_allclose(lower.T@np.diag(d)@lower, PAPER_Q, atol=1e-12)
        self.assertAlmostEqual(d[-1], PAPER_Q[-1, -1])          # bootstrapping starts with the last one


if __name__ == '__main__':
    unittest.main()
