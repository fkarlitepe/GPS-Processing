"""Checks against Teunissen, Joosten & Tiberius (2002), "A comparison of TCAR, CIR and LAMBDA
GNSS ambiguity resolution" (ION GPS 2002)."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from gps_processing.processing.ar.integer import (INTEGER_METHODS, bootstrap_success, evaluate, ldl,
                                                  search, success_rates)
from gps_processing.processing.ml.ar_simulation import narrow_lane_covariance


def from_conditional_sigmas(*sigmas):
    return bootstrap_success(np.array(sigmas)**2)


class PaperTableTests(unittest.TestCase):
    def test_table3_tcar_galileo(self):
        # Short baseline: original DD, TCAR and LAMBDA ambiguities (conditional sigmas, cycles).
        self.assertAlmostEqual(from_conditional_sigmas(1.452, 0.044, 0.039), 0.270, delta=0.002)
        self.assertAlmostEqual(from_conditional_sigmas(0.056, 0.298, 0.151), 0.906, delta=0.002)
        self.assertAlmostEqual(from_conditional_sigmas(0.056, 0.155, 0.290), 0.914, delta=0.002)

    def test_table4_cir_gps(self):
        self.assertAlmostEqual(from_conditional_sigmas(1.360, 0.035, 0.039), 0.287, delta=0.002)
        self.assertAlmostEqual(from_conditional_sigmas(0.068, 0.219, 0.122), 0.978, delta=0.002)
        self.assertAlmostEqual(from_conditional_sigmas(0.068, 0.125, 0.214), 0.980, delta=0.002)

    def test_partial_fixing_numbers(self):
        # Sec. 5.3: first two combinations, long baseline, one satellite pair and three pairs.
        tcar, cir = from_conditional_sigmas(0.060, 0.340), from_conditional_sigmas(0.080, 0.431)
        self.assertAlmostEqual(tcar, 0.8586, places=3)
        self.assertAlmostEqual(cir, 0.7540, places=3)
        self.assertAlmostEqual(tcar**3, 0.6330, places=3)
        self.assertAlmostEqual(cir**3, 0.4287, places=3)

    def test_predefined_transformations_are_admissible(self):
        z_tcar = np.array([[1, -1, 0], [1, 0, -1], [1, 0, 0]])         # eq. (13)
        z_cir = np.array([[0, 1, -1], [1, -1, 0], [0, 0, 1]])          # eq. (14)
        self.assertEqual(round(np.linalg.det(z_tcar)), 1)
        self.assertEqual(round(np.linalg.det(z_cir)), -1)
        self.assertTrue(np.allclose(np.linalg.inv(z_cir), np.rint(np.linalg.inv(z_cir))))

    def test_unit_lower_triangular_keeps_bootstrap_success(self):
        # Sec. 5.1: the long-baseline LAMBDA and TCAR ambiguities differ by a unit lower
        # triangular matrix, which leaves conditional variances (and IB success) unchanged.
        rng = np.random.default_rng(3)
        g = rng.normal(size=(3, 3)); q = g@g.T+0.1*np.eye(3)
        t = np.array([[1, 0, 0], [-3, 1, 0], [-162, 15, 1]], dtype=float)
        _, d = ldl(q)
        _, d_t = ldl(t.T@q@t)       # LAMBDA order conditions from the last element: upper unit form
        self.assertAlmostEqual(bootstrap_success(d_t), bootstrap_success(d), places=10)


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.q = narrow_lane_covariance(np.random.default_rng(5), 7, 8, 0.3, 0.003)

    def test_all_methods_fix_precise_ambiguities(self):
        truth = np.array([3., -2, 5, 0, 7, -4])
        q = self.q/np.max(np.diag(self.q))*1e-3
        for method in INTEGER_METHODS:
            result = evaluate(truth+0.01, q, method, 0.0)
            np.testing.assert_allclose(result['combinations']@truth, result['integers'], err_msg=method)
            self.assertGreater(result['ratio'], 1.0, method)

    def test_success_ordering_ir_ib_ils(self):
        # Teunissen (1999), paper eq. (8) and sec. 2.4: P(IR) <= P(IB) <= P(ILS).
        rates = success_rates(self.q)
        rng = np.random.default_rng(6)
        draws = rng.multivariate_normal(np.zeros(len(self.q)), self.q, 1500)
        hit = {m: np.mean([np.allclose(np.rint(r['integers']), 0) for r in
                           (evaluate(x, self.q, m, 0.0) for x in draws)]) for m in ('cascade', 'bootstrap', 'lambda')}
        noise = 4*math.sqrt(0.25/1500)
        self.assertLessEqual(hit['cascade'], hit['bootstrap']+noise)
        self.assertLessEqual(hit['bootstrap'], hit['lambda']+noise)
        self.assertAlmostEqual(hit['bootstrap'], rates['ib'], delta=noise)
        self.assertLessEqual(rates['ir_original'], hit['cascade']+noise)     # eq. (26) lower bound

    def test_partial_subsets_reach_threshold(self):
        for method in ('lambda', 'bootstrap', 'cascade'):
            result = evaluate(np.zeros(len(self.q)), self.q, method, 0.99)
            if result['dimension'] > 1:
                self.assertGreaterEqual(result['success'], 0.99, method)

    def test_solver_runs_every_method(self):
        from tests.test_ar import ARTests
        from gps_processing.processing.ar.solver import solve
        receiver, obs, sat = ARTests().synthetic()
        cfg = lambda method, success: SimpleNamespace(
            technique='PPP-AR/SDBS', elevation_mask=10, ar_min_epochs=3, ar_ratio=2, ar_max_std=100., estimator='ekf',
            estimate_troposphere=False, physical_corrections=False, robust=True, code_sigma=.3, phase_sigma=.003,
            integer_method=method, ar_success_rate=success, ar_validation='ratio', ml_threshold=.9)
        n1 = {}
        for method in INTEGER_METHODS:
            pos, amb, diag, *_ = solve(obs, sat, receiver, cfg(method, .99 if method != 'cascade' else .8))
            self.assertTrue(pos.status.isin(['fixed', 'partial_fixed']).any(), method)
            wide = amb.dropna(subset=['NW_fixed'])
            np.testing.assert_allclose(wide.NW_fixed, wide.sat-wide.reference_sat, atol=1e-5, err_msg=method)
            n1[method] = diag.n1_success.dropna().iloc[-1]
        # Ignoring the correlation between satellite pairs costs N1 success rate (paper, sec. 5).
        self.assertLess(n1['cascade'], 0.95)
        self.assertGreater(n1['bootstrap'], 0.99)
        self.assertGreater(n1['lambda'], 0.99)
        pos, *_ = solve(obs, sat, receiver, cfg('cascade', .99))
        self.assertFalse(pos.status.isin(['fixed', 'partial_fixed']).any())


if __name__ == '__main__':
    unittest.main()
