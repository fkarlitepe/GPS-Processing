"""SDBS, DCM and integer phase clock (IPC) user models: equivalence as in Shi & Gao (2014)."""
import unittest
from types import SimpleNamespace
import numpy as np
from gps_processing.processing.ar.ipc import WideLaneResolver
from gps_processing.processing.ar.models import LAMBDA_W
from gps_processing.processing.ar.solver import solve
from tests import test_ar

METHODS = ('SDBS', 'DCM', 'IPC')


def config(method, **kw):
    base = dict(technique='PPP-AR/'+method, elevation_mask=10, ar_min_epochs=3, ar_ratio=3, ar_max_std=.5,
                estimator='ekf', estimate_troposphere=False, physical_corrections=False, robust=True,
                code_sigma=.3, phase_sigma=.003, integer_method='lambda', ar_success_rate=.99, ar_validation='ratio')
    base.update(kw)
    return SimpleNamespace(**base)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.receiver, self.obs, self.sat = test_ar.ARTests().synthetic()
        self.start = self.receiver+np.array([2., -3., 4.])

    def test_three_models_are_equivalent(self):
        finals = {}
        for method in METHODS:
            pos, amb, diag, *_ = solve(self.obs, self.sat, self.start, config(method))
            last = pos.iloc[-1]
            self.assertIn(last.status, ('fixed', 'partial_fixed'), method)
            finals[method] = last[['x', 'y', 'z']].to_numpy(float)
            self.assertLess(np.linalg.norm(finals[method]-self.receiver), 0.02, method)
            fixed = amb.dropna(subset=['NW_fixed', 'N1_fixed'])
            ref = fixed.reference_sat
            # Synthetic truth: N1 = 3*sat, NW = sat-2, so SD ambiguities are 3*(j-r) and (j-r).
            np.testing.assert_allclose(fixed.NW_fixed, fixed.sat-ref, atol=1e-5, err_msg=method)
            np.testing.assert_allclose(fixed.N1_fixed, 3*(fixed.sat-ref), atol=1e-5, err_msg=method)
        for method in METHODS[1:]:
            np.testing.assert_allclose(finals[method], finals['SDBS'], atol=0.01, err_msg=method)

    def test_ipc_wide_lane_receiver_bias(self):
        # Synthetic MW carries a 3 m receiver bias: WRB fraction = frac(3/lambda_W) (eq. 46).
        pos, _, diag, *_ = solve(self.obs, self.sat, self.start, config('IPC'))
        fraction = (3/LAMBDA_W+0.5) % 1-0.5
        self.assertAlmostEqual(pos.receiver_mw_bias_m.iloc[-1]/LAMBDA_W, fraction, places=6)
        self.assertEqual(diag.wl_known.iloc[-1], 6)

    def test_reference_change_keeps_converged_ambiguities(self):
        pos, *_ = solve(self.obs, self.sat, self.start, config('DCM'))
        first_ref = int(pos.reference_sat.iloc[0])
        cut = self.obs[~((self.obs.sat == first_ref) & (self.obs.epok >= 450))]
        for method in METHODS:
            pos, amb, diag, *_ = solve(cut, self.sat, self.start, config(method))
            after = pos[pos.epok >= 450]
            self.assertNotEqual(int(after.reference_sat.iloc[0]), first_ref)
            # Without a reset the next epoch is still fixed (a reset would need ar_min_epochs again).
            self.assertIn(after.status.iloc[0], ('fixed', 'partial_fixed'), method)
            self.assertLess(np.linalg.norm(after.iloc[-1][['x', 'y', 'z']].to_numpy(float)-self.receiver), 0.02)
            fixed = amb[(amb.epok >= 450)].dropna(subset=['N1_fixed'])
            np.testing.assert_allclose(fixed.N1_fixed, 3*(fixed.sat-fixed.reference_sat), atol=1e-5, err_msg=method)

    def test_resolver_gates(self):
        resolver = WideLaneResolver(min_epochs=3)
        rng = np.random.default_rng(0)
        for _ in range(10):
            for sat, n in ((1, 4), (2, -7), (3, 12)):
                resolver.add(sat, LAMBDA_W*(n+0.45+rng.normal(0, 0.02)))
            resolver.add(4, LAMBDA_W*(2+0.45+rng.normal(0, 0.6)))         # noisy arc stays unresolved
        result = resolver.resolve([1, 2, 3, 4])
        self.assertEqual(result, {1: 4, 2: -7, 3: 12})
        self.assertAlmostEqual(resolver.receiver_bias, 0.45, delta=0.02)


if __name__ == '__main__':
    unittest.main()
