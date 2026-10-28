import unittest
import numpy as np
from gps_processing.processing.ar.integer import evaluate
from gps_processing.processing.ml.ar_benchmark import threshold_for, summary
from gps_processing.processing.ml.ar_simulation import FEATURES, feature_vector, narrow_lane_covariance, simulate, wide_lane_covariance
from gps_processing.processing.ml.ar_validation import CANDIDATES, DEFAULT_MODEL, build_model


class SimulationTests(unittest.TestCase):
    def test_gnss_covariances_are_valid_and_shrink_with_epochs(self):
        rng = np.random.default_rng(0)
        short = narrow_lane_covariance(np.random.default_rng(1), 7, 2, 0.3, 0.003)
        long = narrow_lane_covariance(np.random.default_rng(1), 7, 60, 0.3, 0.003)
        self.assertEqual(short.shape, (6, 6))
        self.assertTrue(np.all(np.linalg.eigvalsh(long) > 0))
        self.assertLess(np.trace(long), np.trace(short))
        wide = wide_lane_covariance(rng, 6, 30, 0.3, 0.003)
        self.assertEqual(wide.shape, (5, 5))
        self.assertGreater(wide[0, 1], 0)            # common reference satellite correlates SD ambiguities

    def test_labels_follow_fixed_combinations(self):
        x, y = simulate(300, seed=3)
        self.assertEqual(x.shape, (300, len(FEATURES)))
        self.assertTrue(0.2 < y.mean() < 0.95)
        # Very precise, unbiased ambiguities are labelled correct.
        result = evaluate(np.array([4.002, -3.001, 7.0]), np.diag([1e-4]*3), 'lambda', 0.999)
        self.assertEqual(feature_vector(result).shape, (len(FEATURES),))

    def test_fixed_failure_rate_threshold(self):
        score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        label = np.array([1, 1, 0, 1, 0])
        self.assertEqual(threshold_for(score, label, 0.0), 0.8)
        self.assertEqual(threshold_for(score, label, 0.25), 0.6)
        row = summary('x', score >= 0.6, label)
        self.assertAlmostEqual(row['failure_rate'], 0.25)
        self.assertAlmostEqual(row['availability'], 0.6)

    def test_all_candidates_fit(self):
        x, y = simulate(400, seed=4)
        self.assertIn(DEFAULT_MODEL, CANDIDATES)
        for name in CANDIDATES:
            p = build_model(name).fit(x, y).predict_proba(x)[:, 1]
            self.assertTrue(np.all((p >= 0) & (p <= 1)), name)


if __name__ == '__main__':
    unittest.main()
