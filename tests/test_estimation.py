import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from gps_processing.constants import C
from gps_processing.analysis import add_enu, enu_rotation, convergence_time
from gps_processing.processing.estimation import ExtendedKalmanFilter, igg3
from gps_processing.processing.troposphere import Troposphere, geodetic
from gps_processing.processing.ar.integer import decorrelate, evaluate, bootstrap_success, search
from gps_processing.processing.ml.learners import GradientBoostingRegressor, LogisticRegression
from gps_processing.processing.ml.stochastic import ElevationModel, LearnedVarianceModel, IF_FACTOR
from gps_processing.processing import ppp_ekf

ROOT = Path(__file__).resolve().parents[1]
RECEIVER = np.array([4849202.3940, -360328.9929, 4114913.1862])


def sky(receiver, epochs=40, sats=8, tropo_zwd=0.0, outlier=None):
    """Well-spread GPS-like geometry; IF code/phase with clock, ZHD+ZWD and constant ambiguities."""
    rotation = enu_rotation(receiver)
    troposphere = Troposphere(receiver)
    rows, positions = [], []
    for epoch in range(epochs):
        for sat in range(1, sats+1):
            azimuth = 2*np.pi*sat/sats+epoch*2e-3
            elev = np.radians(20+60*((sat*37) % 7)/6)+epoch*1e-3
            enu = np.array([np.cos(elev)*np.sin(azimuth), np.cos(elev)*np.cos(azimuth), np.sin(elev)])
            xyz = receiver+rotation.T@enu*21e6
            distance = np.linalg.norm(xyz-receiver)
            dry, wet = troposphere.mapping(np.degrees(elev))
            delay = troposphere.zhd*dry+tropo_zwd*wet
            pc = distance+30+delay
            if outlier and outlier == (epoch, sat):
                pc += 80
            rows.append(dict(epok=epoch*30., sat=sat, Pc=pc, Lc=distance+30+delay+0.1*sat, A4=0., GF=0.,
                             epoch_reset=False, dt_sat=0., snr1=45., snr2=40.))
            positions.append(dict(epok=epoch*30., sat=sat, x=xyz[0], y=xyz[1], z=xyz[2], relativity_m=0.))
    return pd.DataFrame(rows), pd.DataFrame(positions)


def config(**kw):
    base = dict(estimator='ekf', dynamics='static', position_noise=10., estimate_troposphere=True, troposphere_noise=5.,
                physical_corrections=True, robust=True, code_sigma=.3, phase_sigma=.003, elevation_mask=10)
    base.update(kw)
    return SimpleNamespace(**base)


class EKFTests(unittest.TestCase):
    def test_linear_update_matches_least_squares(self):
        h = np.array([[1., 0], [0, 1], [1, 1]]); z = np.array([1., 2, 3.3]); r = np.diag([.1, .2, .3])
        kf = ExtendedKalmanFilter(np.zeros(2), np.eye(2)*1e12)
        kf.update(lambda x, inflation: (z, h@x, h, r*inflation[[0, 1, 2]], np.arange(3)), 3)
        w = np.linalg.inv(r)
        np.testing.assert_allclose(kf.x, np.linalg.solve(h.T@w@h, h.T@w@z), atol=1e-6)
        np.testing.assert_allclose(kf.P, np.linalg.inv(h.T@w@h), atol=1e-6)

    def test_igg3_segments(self):
        np.testing.assert_allclose(igg3(np.array([1., 2.5])), 1)
        self.assertGreater(igg3(np.array([4.]))[0], 1)
        self.assertGreater(igg3(np.array([7.]))[0], 1e6)

    def test_float_ppp_recovers_position_clock_and_zwd(self):
        obs, sat = sky(RECEIVER, tropo_zwd=0.15)
        out = ppp_ekf.solve(obs, sat, RECEIVER+np.array([5., -4, 6]), config())
        last = out.positions.iloc[-1]
        np.testing.assert_allclose(last[['x', 'y', 'z']].to_numpy(float), RECEIVER, atol=0.02)
        self.assertAlmostEqual(last.receiver_clock_s*C, 30, delta=0.05)
        self.assertAlmostEqual(last.zwd_m, 0.15, delta=0.01)
        self.assertEqual(len(out.residuals), len(obs))
        self.assertGreaterEqual(np.linalg.eigvalsh(out.covariance).min(), -1e-6)

    def test_robust_update_rejects_code_outlier(self):
        obs, sat = sky(RECEIVER, epochs=3, outlier=(0, 3))
        start = RECEIVER+np.array([3., 2, -1])
        plain = ppp_ekf.solve(obs, sat, start, config(robust=False)).positions.iloc[0]
        robust = ppp_ekf.solve(obs, sat, start, config(robust=True))
        error = lambda p: np.linalg.norm(p[['x', 'y', 'z']].to_numpy(float)-RECEIVER)
        self.assertLess(error(robust.positions.iloc[0]), error(plain)/5)
        self.assertGreater(robust.residuals.query('epok==0 and sat==3').inflation.iloc[0], 10)

    def test_kinematic_follows_motion(self):
        obs, sat = sky(RECEIVER, epochs=20)
        moved = obs.copy()
        shift = np.where(moved.epok >= 300, 2.0, 0.0)
        # A 2 m vertical jump at epoch 10 projected on each line of sight.
        up = enu_rotation(RECEIVER)[2]
        los = (sat[['x', 'y', 'z']].to_numpy()-RECEIVER)
        los /= np.linalg.norm(los, axis=1)[:, None]
        moved['Pc'] -= shift*(los@up); moved['Lc'] -= shift*(los@up)
        static = ppp_ekf.solve(moved, sat, RECEIVER, config()).positions
        kinematic = ppp_ekf.solve(moved, sat, RECEIVER, config(dynamics='kinematic', position_noise=5.)).positions
        up_error = lambda p: (p[['x', 'y', 'z']].iloc[-1].to_numpy(float)-RECEIVER)@up
        self.assertAlmostEqual(up_error(kinematic), 2.0, delta=0.05)
        self.assertLess(up_error(static), 1.5)

    def test_troposphere_model(self):
        lat, lon, height = geodetic(RECEIVER)
        self.assertAlmostEqual(np.degrees(lat), 40.43, delta=0.05)
        self.assertGreater(height, 0)
        t = Troposphere(RECEIVER)
        self.assertTrue(2.0 < t.zhd < 2.4 and 0.05 < t.zwd < 0.3)
        dry, wet = t.mapping(np.array([90., 10.]))
        self.assertAlmostEqual(dry[0], 1, places=2)
        self.assertTrue(5 < dry[1] < 6)

    def test_niell_mapping(self):
        from gps_processing.processing.troposphere import chao_mapping, niell_mapping
        lat = np.radians(40.43)
        dry, wet = niell_mapping(np.array([90., 5.]), lat, 0.0, 180)
        np.testing.assert_allclose([dry[0], wet[0]], 1.0, atol=1e-12)
        self.assertAlmostEqual(dry[1], 10.15, delta=0.06)              # textbook NMF values at 5 deg
        self.assertAlmostEqual(wet[1], 10.75, delta=0.06)
        # Agrees with Chao above 15 deg; annual term flips with the hemisphere; height raises the dry map.
        np.testing.assert_allclose(niell_mapping(np.array([20., 45.]), lat, 0.0)[0], chao_mapping(np.array([20., 45.]))[0], rtol=2e-3)
        self.assertAlmostEqual(niell_mapping(7., -lat, 0.0, 28)[0], niell_mapping(7., lat, 0.0, 28+182.625)[0], places=12)
        self.assertNotAlmostEqual(niell_mapping(7., lat, 0.0, 28)[0], niell_mapping(7., lat, 0.0, 210)[0], places=4)
        self.assertGreater(niell_mapping(7., lat, 2000.0)[0], niell_mapping(7., lat, 0.0)[0])
        self.assertEqual(Troposphere(RECEIVER, mapping='chao').mapping(10.)[0], chao_mapping(10.)[0])
        with self.assertRaises(ValueError):
            Troposphere(RECEIVER, mapping='gmf')


class AmbiguityTests(unittest.TestCase):
    def test_lambda_matches_brute_force(self):
        rng = np.random.default_rng(4)
        for _ in range(5):
            lower = np.tril(rng.normal(size=(4, 4)), -1)+np.eye(4)
            q = lower@np.diag(rng.uniform(.02, .2, 4))@lower.T
            a = rng.normal(size=4)*2
            z, l, d = decorrelate(q)
            self.assertAlmostEqual(abs(np.linalg.det(z)), 1)
            np.testing.assert_allclose(l.T@np.diag(d)@l, z.T@q@z, atol=1e-9)
            result = evaluate(a, q, 'lambda', 0.0)
            fixed = np.rint(np.linalg.solve(result['combinations'], result['integers']))
            inv = np.linalg.inv(q)
            best = min(itertools.product(*[range(int(np.floor(v))-3, int(np.ceil(v))+4) for v in a]),
                       key=lambda c: (a-np.array(c))@inv@(a-np.array(c)))
            np.testing.assert_equal(fixed, best)

    def test_bootstrap_success_is_lower_bound(self):
        q = np.array([[.04, .03], [.03, .05]])
        z, _, d = decorrelate(q)
        bound = bootstrap_success(d)
        rng = np.random.default_rng(0)
        draws = rng.multivariate_normal(np.zeros(2), q, 4000)
        hits = sum(np.array_equal(search(a, q)[0][0], [0, 0]) for a in draws)
        self.assertLessEqual(bound, hits/4000+0.02)
        self.assertGreater(bound, 0.5)

    def test_partial_ar_keeps_precise_subset(self):
        q = np.diag([1e-4, 2e-4, 0.5, 0.8])
        result = evaluate(np.array([3.01, -2.0, 0.4, 1.3]), q, 'lambda', 0.999)
        self.assertEqual(result['dimension'], 2)
        self.assertGreaterEqual(result['success'], 0.999)
        full = evaluate(np.array([3.01, -2.0, 0.4, 1.3]), q, 'ils', 0.999)
        self.assertEqual(full['dimension'], 4)


class LearningTests(unittest.TestCase):
    def test_gradient_boosting_learns_nonlinear_function(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, (3000, 2)); y = np.sin(6*x[:, 0])+x[:, 1]**2
        model = GradientBoostingRegressor(n_estimators=80).fit(x, y)
        self.assertLess(np.sqrt(np.mean((model.predict(x)-y)**2)), 0.08)
        clone = GradientBoostingRegressor.from_dict(model.to_dict())
        np.testing.assert_allclose(clone.predict(x[:10]), model.predict(x[:10]))

    def test_logistic_regression_separates(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(500, 2)); y = (x[:, 0]+0.5*x[:, 1] > 0).astype(float)
        model = LogisticRegression().fit(x, y)
        self.assertGreater(np.mean((model.probability(x) > .5) == y), 0.95)

    def test_learned_variance_tracks_elevation(self):
        rng = np.random.default_rng(3)
        elev = rng.uniform(10, 90, 4000)
        sine = np.sin(np.radians(elev))
        residuals = pd.DataFrame(dict(elevation=elev, snr1=30+20*sine, snr2=25+20*sine, arc_epochs=50,
                                      code_res=rng.normal(size=4000)*IF_FACTOR*0.8/sine,
                                      phase_res=rng.normal(size=4000)*IF_FACTOR*0.004/sine))
        model = LearnedVarianceModel.fit(residuals, ElevationModel(.3, .003))
        probe = pd.DataFrame(dict(elevation=[15., 80.], snr1=[35., 49.], snr2=[30., 44.], arc_epochs=[50, 50]))
        code, phase = model.sigmas(probe)
        self.assertGreater(code[0], 2*code[1])
        self.assertAlmostEqual(code[1], 0.8/np.sin(np.radians(80)), delta=0.2)
        self.assertAlmostEqual(phase[1], 0.004/np.sin(np.radians(80)), delta=0.001)

    def test_validator_probability_orders_quality_and_round_trips(self):
        from gps_processing.processing.ml.ar_validation import ARValidator
        validator = ARValidator.train(samples=1500, seed=1)
        from gps_processing.processing.ml.ar_simulation import add_policy_features
        # The solver adds the FFRT threshold, P0 and fraction features before asking the validator.
        good_a, bad_a = np.array([1.01, 2.0, -0.98]), np.array([1.45, 2.4, -0.6])
        good = add_policy_features(evaluate(good_a, np.diag([1e-3]*3), 'lambda', 0.0), good_a, np.diag([1e-3]*3))
        bad = add_policy_features(evaluate(bad_a, np.diag([0.3]*3), 'lambda', 0.0), bad_a, np.diag([0.3]*3))
        self.assertGreater(validator.probability(good), 0.95)
        self.assertLess(validator.probability(bad), validator.probability(good))
        with tempfile.TemporaryDirectory() as directory:
            validator.save(Path(directory)/'v.joblib')
            loaded = ARValidator.load(Path(directory)/'v.joblib')
            with self.assertRaises(ValueError):
                ARValidator.load(Path(directory)/'v.json')
            self.assertAlmostEqual(loaded.probability(good), validator.probability(good))


class ARFilterTests(unittest.TestCase):
    def test_ekf_lambda_and_ml_validation_fix_correct_integers(self):
        from tests.test_ar import ARTests
        from gps_processing.processing.ar.solver import solve
        receiver, obs, sat = ARTests().synthetic()
        for method in ('SDBS', 'DCM'):
            for validation in ('ratio', 'ratio+ml'):
                cfg = SimpleNamespace(technique='PPP-AR/'+method, elevation_mask=10, ar_min_epochs=3, ar_ratio=3, ar_max_std=100.,
                                      estimator='ekf', estimate_troposphere=False, physical_corrections=False, robust=True,
                                      code_sigma=.3, phase_sigma=.003, integer_method='lambda', ar_success_rate=.99,
                                      ar_validation=validation, ml_threshold=.9)
                pos, amb, diag, *_ = solve(obs, sat, receiver, cfg)
                self.assertTrue(pos.status.isin(['fixed', 'partial_fixed']).any(), (method, validation))
                # PAR may fix decorrelated N1 combinations only; individual N1 are then not integer.
                wide = amb.dropna(subset=['NW_fixed'])
                self.assertFalse(wide.empty)
                np.testing.assert_allclose(wide.NW_fixed, wide.sat-wide.reference_sat)
                narrow = amb.dropna(subset=['N1_fixed'])
                np.testing.assert_allclose(narrow.N1_fixed, 3*(narrow.sat-narrow.reference_sat))
                last = pos.iloc[-1]
                self.assertLess(np.linalg.norm(last[['x', 'y', 'z']].to_numpy(float)-receiver), 0.05)
                if 'ml' in validation:
                    self.assertTrue(diag.n1_probability.notna().any())


class AnalysisAndPipelineTests(unittest.TestCase):
    def test_enu_and_convergence(self):
        up = enu_rotation(RECEIVER)[2]
        frame = pd.DataFrame(dict(epok=[0., 30, 60, 90], status='float', satellites=8))
        frame[['x', 'y', 'z']] = RECEIVER+np.outer([1., .3, .05, .02], up)
        enu = add_enu(frame, RECEIVER)
        np.testing.assert_allclose(enu.up, [1, .3, .05, .02], atol=1e-9)
        np.testing.assert_allclose(enu[['east', 'north']], 0, atol=1e-9)
        self.assertEqual(convergence_time(enu), 60)

    def test_config_validation_of_new_options(self):
        from gps_processing.models import ProcessingConfig
        for kw, error in [(dict(estimator='ukf'), 'Kestirici'), (dict(estimator='sequential', ml_weighting=True), 'ML'),
                          (dict(ar_success_rate=1.5), 'Başarı'), (dict(reference_xyz=(1, 2, 3)), 'Referans')]:
            with self.assertRaisesRegex(ValueError, error):
                ProcessingConfig('a', 'b', 'c', 'd', **kw).validate()

    @unittest.skipUnless(len(list((ROOT/'Data').glob('*.SP3'))) == 3, 'örnek veri yok')
    def test_sample_comparison(self):
        from gps_processing.models import ProcessingConfig
        from gps_processing.pipeline import compare
        sp3 = sorted((ROOT/'Data').glob('*.SP3'))
        with tempfile.TemporaryDirectory() as directory:
            sample = sorted([*(ROOT/'Data').glob('*.rnx'), *(ROOT/'Data').glob('*.??o')])[0]
            cfg = ProcessingConfig(*sp3, sample, output=Path(directory),
                                   reference_xyz=tuple(RECEIVER))
            table, results = compare(cfg, ['Sequential LS (v2.0)', 'EKF', 'EKF + ML weighting', 'AR · LAMBDA + PAR'])
            self.assertEqual(list(table.solution), ['Sequential LS (v2.0)', 'EKF', 'EKF + ML weighting'])
            folder = table.attrs['output']                      # <output>/<year>_<doy>_<HHMMSS>
            self.assertEqual(folder.parent, Path(directory))
            self.assertTrue((folder/'comparison'/'statistics.csv').is_file())
            self.assertTrue((folder/'comparison'/'ekf_ml_weighting'/'ml_models.json').is_file())
            self.assertLess(table.set_index('solution').loc['EKF', 'final_3d'], table.set_index('solution').loc['Sequential LS (v2.0)', 'final_3d'])


if __name__ == '__main__':
    unittest.main()


class QualityControlTests(unittest.TestCase):
    """Global chi-square test, Baarda data snooping, Pope tau test, Huber and Danish robust estimation."""

    def setUp(self):
        from gps_processing.processing.estimation import quality
        self.q = quality
        rng = np.random.default_rng(3)
        self.t = np.linspace(0, 10, 25)
        self.A = np.column_stack([np.ones_like(self.t), self.t])
        self.truth = np.array([2.0, 0.5])
        self.l = self.A@self.truth+rng.normal(0, 0.01, len(self.t))
        self.Q = np.eye(len(self.t))*0.01**2

    def test_critical_values(self):
        q = self.q
        self.assertAlmostEqual(q.baarda_lambda(0.001, 0.80), 17.07, delta=0.01)
        # one observation: the B-method global test is the local two-sided w-test
        self.assertAlmostEqual(q.b_method_alpha(1, 0.001, 0.80), 0.001, delta=2e-4)
        self.assertGreater(q.b_method_alpha(20, 0.001, 0.80), 0.001)
        # tau tends to the normal quantile for many degrees of freedom and is bounded by sqrt(df)
        self.assertAlmostEqual(q.tau_critical(5000, 0.001), 3.29, delta=0.01)
        self.assertLess(q.tau_critical(4, 0.05), 2.0)

    def test_weight_functions(self):
        q = self.q
        np.testing.assert_allclose(q.huber(np.array([1.0, 3.0]), 1.5), [1.0, 2.0])
        np.testing.assert_allclose(q.danish(np.array([1.9, 2.0])), 1.0)
        self.assertAlmostEqual(1/q.danish(np.array([4.0]), 2.0)[0], np.exp(-3), places=10)
        np.testing.assert_allclose(q.robust_inflation('igg3', np.array([7.])), 1e8)

    def test_global_test_size(self):
        """False-alarm rate of the chi-square test without outliers is close to its level."""
        rng = np.random.default_rng(1)
        options = self.q.QualityOptions(test='global', global_alpha=0.05)
        rejected = [not self.q.least_squares(self.A, self.A@self.truth+rng.normal(0, .01, len(self.t)), self.Q,
                                             options).test.passed for _ in range(800)]
        self.assertAlmostEqual(np.mean(rejected), 0.05, delta=0.025)

    def test_baarda_and_pope_identify_blunder(self):
        l = self.l.copy(); l[7] += 0.12
        for test in ('baarda', 'pope'):
            out = self.q.least_squares(self.A, l, self.Q, self.q.QualityOptions(test=test))
            self.assertFalse(out.test.first_passed, test)
            self.assertEqual(out.test.excluded[0], 7, test)   # Baarda also rejects obs. 9 (w = 3.5 without the blunder)
            self.assertTrue(out.test.passed, test)
            np.testing.assert_allclose(out.x, self.truth, atol=0.01)
        flagged = self.q.least_squares(self.A, l, self.Q, self.q.QualityOptions(test='global'))
        self.assertFalse(flagged.test.passed)
        self.assertEqual(flagged.test.excluded, [])

    def test_huber_and_danish_reduce_blunder_influence(self):
        l = self.l.copy(); l[3] += 0.5
        plain = self.q.least_squares(self.A, l, self.Q)
        for method in ('huber', 'danish', 'igg3'):
            out = self.q.least_squares(self.A, l, self.Q, robust=method)
            self.assertLess(np.abs(out.x-self.truth).max(), np.abs(plain.x-self.truth).max()/5, method)
            self.assertLess(out.weights[3], 0.2, method)

    def test_filter_tests_exclude_code_outlier(self):
        obs, sat = sky(RECEIVER, epochs=3, outlier=(1, 3))
        start = RECEIVER+np.array([3., 2, -1])
        error = lambda p: np.linalg.norm(p[['x', 'y', 'z']].to_numpy(float)-RECEIVER)
        plain = ppp_ekf.solve(obs, sat, start, config(robust=False)).positions.iloc[1]
        for test in ('baarda', 'pope'):
            out = ppp_ekf.solve(obs, sat, start, config(robust=False, outlier_test=test)).positions
            row = out.iloc[1]
            self.assertFalse(row.qc_first_passed, test)
            self.assertEqual(str(row.qc_excluded_sats), '3', test)
            self.assertLess(error(row), error(plain)/5, test)
            self.assertTrue(out.iloc[[0, 2]].qc_first_passed.all(), test)
        for method in ('huber', 'danish'):
            row = ppp_ekf.solve(obs, sat, start, config(robust=True, robust_method=method)).positions.iloc[1]
            self.assertLess(error(row), error(plain)/5, method)

    def test_defaults_unchanged(self):
        """No test and IGG-III: identical to the update without quality options."""
        obs, sat = sky(RECEIVER, epochs=5, outlier=(2, 4))
        a = ppp_ekf.solve(obs, sat, RECEIVER, config()).positions
        b = ppp_ekf.solve(obs, sat, RECEIVER, config(outlier_test='none', robust_method='igg3')).positions
        np.testing.assert_allclose(a[['x', 'y', 'z']].to_numpy(float), b[['x', 'y', 'z']].to_numpy(float))
        self.assertNotIn('qc_T', a.columns)
