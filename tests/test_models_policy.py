"""v2.9: site models, ORBEX attitude, AR policy (P0, FFRT, exclusion), configuration files,
header-based product checks, vectorized interpolation and the learned fixability."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from gps_processing.processing import tides
from gps_processing.processing.ar.integer import fix_probability, ffrt_threshold

DAY = datetime(2023, 11, 23)
SITE = np.array([4849202.176, -360328.543, 4114913.418])


def write(directory, name, text):
    path = Path(directory)/name
    path.write_text(text)
    return path


class SiteModelTests(unittest.TestCase):
    def test_blq_ocean_loading_m2(self):
        from gps_processing.io.blq import read_blq, select_station
        block = np.zeros((6, 11))
        block[0, 0] = 0.01                                   # 1 cm M2 radial
        lines = ['$$ Onsala ocean loading', '  MADR', '$$ comment'] + ['  '+' '.join(f'{v:.5f}' for v in row) for row in block]
        with tempfile.TemporaryDirectory() as directory:
            stations = read_blq(write(directory, 'site.blq', '\n'.join(lines)+'\n'))
        coefficients = select_station(stations, 'MADR 13407S012')
        epochs = np.arange(0, 2*86400, 600.0)
        up = tides.ocean_loading(coefficients, SITE, epochs, DAY)@(SITE/np.linalg.norm(SITE))
        self.assertAlmostEqual(np.abs(up).max(), 0.01, places=4)
        spectrum = np.abs(np.fft.rfft(up))
        period_h = 1/np.fft.rfftfreq(len(up), 600.0)[np.argmax(spectrum[1:])+1]/3600
        self.assertAlmostEqual(period_h, 12.42, delta=0.6)
        with self.assertRaises(ValueError):
            select_station(stations, 'ANKR')

    def test_pole_tide_and_shapiro_magnitudes(self):
        from gps_processing.io.erp import read_erp, pole
        text = 'version 2\n  MJD  Xpole Ypole\n 60270.50  150000  380000 0\n 60271.50  151000  381000 0\n'
        with tempfile.TemporaryDirectory() as directory:
            erp = read_erp(write(directory, 'x.erp', text))
        xp, yp = pole(erp, DAY, np.array([0.0, 43200.0]))
        self.assertAlmostEqual(xp[1], 0.151, places=6)        # MJD 60271.5 = 2023-11-23 12:00
        self.assertAlmostEqual(xp[0], 0.1505, places=6)
        displacement = tides.pole_tide(SITE, xp, yp, DAY)
        self.assertTrue(1e-4 < np.linalg.norm(displacement[0]) < 0.025)
        zenith = SITE+SITE/np.linalg.norm(SITE)*20.2e6
        delay = tides.shapiro(np.array([zenith]), SITE)[0]
        self.assertTrue(0.012 < delay < 0.0135)

    def test_site_models_change_code_and_phase_equally(self):
        combined = pd.DataFrame(dict(epok=[0.0, 0.0], sat=[1, 2], Pc=[2e7, 2.1e7], Lc=[2e7, 2.1e7]))
        satellites = pd.DataFrame(dict(epok=[0.0, 0.0], sat=[1, 2], x=[2e7, 1e7], y=[1e7, -1.5e7], z=[1.5e7, 2e7]))
        out = tides.site_models(combined, satellites, SITE, DAY, solid=True, relativity=True)
        np.testing.assert_allclose(out.Pc-combined.Pc, out.Lc-combined.Lc)
        self.assertTrue({'tide_m', 'shapiro_m'} <= set(out))


class AttitudeTests(unittest.TestCase):
    def test_orbex_identity_and_interpolation(self):
        from gps_processing.io.orbex import read_orbex, axes
        text = ('%=ORBEX 0.09\n+FILE/DESCRIPTION\n FRAME_TYPE          ECEF\n-FILE/DESCRIPTION\n+EPHEMERIS/DATA\n'
                '## 2023 11 23  0  0  0.000000000000 1\n ATT G05              4  1.0 0.0 0.0 0.0\n'
                '## 2023 11 23  0  5  0.000000000000 1\n ATT G05              4  0.0 0.0 0.0 1.0\n-EPHEMERIS/DATA\n')
        with tempfile.TemporaryDirectory() as directory:
            attitude = read_orbex(write(directory, 'a.obx', text), DAY)
        x, y, z, covered = axes(attitude, np.array([5, 5, 7]), np.array([0.0, 150.0, 0.0]))
        np.testing.assert_allclose(x[0], [1, 0, 0]); np.testing.assert_allclose(z[0], [0, 0, 1])
        self.assertTrue(covered[1] and not covered[2])
        np.testing.assert_allclose(np.linalg.norm(x[1]), 1.0)        # halfway: 90 deg yaw about z
        np.testing.assert_allclose(np.abs(x[1]), [0, 1, 0], atol=1e-12)


class PolicyTests(unittest.TestCase):
    def test_dong_bock_probability(self):
        p0 = fix_probability(np.array([3.0, 3.02, 3.4, 3.4]), np.array([0.3, 0.03, 0.03, 0.3]))
        self.assertAlmostEqual(p0[0], 1.0)
        self.assertGreater(p0[1], 0.999)
        self.assertGreater(p0[2], 0.999)                 # P0 is driven by sigma; b matters as sigma grows
        self.assertLess(p0[3], 0.99)

    def test_ffrt_threshold_grows_with_weak_models(self):
        self.assertEqual(ffrt_threshold(np.diag([1e-3]*4)), 1.0)
        weak = ffrt_threshold(np.diag([0.09]*6), samples=4000)
        weaker = ffrt_threshold(np.diag([0.16]*6), samples=4000)
        self.assertGreater(weak, 1.0)
        self.assertGreater(weaker, weak)

    def settings(self, **kw):
        base = dict(method='lambda', ratio=3.0, max_std=0.15, success=0.999, validation='ffrt', threshold=0.99,
                    failure_rate=0.001, min_ambiguities=3, max_exclusions=2, gates=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_exclusion_drops_the_biased_satellite(self):
        from gps_processing.processing.ar.policy import run_stage
        truth = np.array([3., -2., 5., 1., 7.])
        state = truth+np.array([0.01, -0.02, 0.0, 0.14, 0.01])     # the fourth is biased
        cov = np.diag([0.05**2]*5)
        stage, result, reason, used = run_stage(state, cov, list(range(5)), self.settings(), None, 'n1')
        self.assertIsNotNone(stage, reason)
        self.assertNotIn(3, used[:result['dimension']] if result['excluded'] else [])
        fixed = stage[0]
        np.testing.assert_allclose(fixed[[0, 1, 2, 4]], truth[[0, 1, 2, 4]], atol=1e-6)

    def test_too_few_candidates_and_legacy_settings(self):
        from gps_processing.processing.ar.policy import run_stage
        state, cov = np.array([0.45, 1.4, 2.0]), np.diag([0.01]*3)
        _, _, reason, _ = run_stage(state, cov, [0, 1, 2], self.settings(), None, 'n1')
        self.assertEqual(reason, 'n1_too_few_candidates')
        stage, *_ = run_stage(np.array([1.01, 2.0, -0.99]), np.diag([1e-4]*3), [0, 1, 2],
                              self.settings(validation='ratio', gates=False, min_ambiguities=2, max_exclusions=0), None, 'n1')
        self.assertIsNotNone(stage)


class ConfigurationTests(unittest.TestCase):
    def test_round_trip_and_unknown_keys(self):
        from gps_processing.models import ProcessingConfig
        config = ProcessingConfig('', 'a.sp3', '', 'b.23o', systems=('G', 'E'), ar_validation='ffrt', reference_xyz=(1., 2., 3.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run.json'
            config.to_file(path)
            loaded = ProcessingConfig.from_file(path, elevation_mask=12.0)
            self.assertEqual(loaded.systems, ('G', 'E'))
            self.assertEqual(loaded.reference_xyz, (1., 2., 3.))
            self.assertEqual(loaded.elevation_mask, 12.0)
            bad = write(directory, 'bad.json', json.dumps({'sp3_day': 'x', 'unknown_key': 1}))
            with self.assertRaisesRegex(ValueError, 'unknown_key'):
                ProcessingConfig.from_file(bad)


class ProductHeaderTests(unittest.TestCase):
    def test_renamed_bias_file_is_detected_by_its_header(self):
        from gps_processing.pipeline import product_consistency, owner
        with tempfile.TemporaryDirectory() as directory:
            bia = write(directory, 'GBM0MGXRAP_20233270000_01D_30S_ABS.BIA', '%=BIA 1.00 CNT 2023:333:49359 CNT 2023:327:00000\n')
            clk = write(directory, 'GBM0MGXRAP_20233270000_01D_30S_CLK.CLK',
                        '     3.00           C                                       RINEX VERSION / TYPE\n'
                        'GFZ  GeoForschungsZentrum Potsdam                           ANALYSIS CENTER\n')
            self.assertEqual((owner(bia), owner(clk)), ('CNES', 'GFZ'))
            config = SimpleNamespace(technique='PPP-AR/SDBS', ar_products=str(bia), clock_file=str(clk),
                                     sp3_day='GBM0MGXRAP_20233270000_01D_05M_ORB.SP3')
            errors, _ = product_consistency(config)
            self.assertTrue(any('header says' in e for e in errors))
            self.assertTrue(any('CNES' in e and 'CLK' in e for e in errors))


class InterpolationTests(unittest.TestCase):
    def test_vectorized_lagrange_matches_scalar(self):
        from gps_processing.processing.orbit import interpolate, lagrange
        nodes = np.arange(0, 86400+1, 900.0)
        values = np.column_stack([np.sin(nodes/7000), np.cos(nodes/9000), nodes/1e5])*2e4
        targets = np.array([0.0, 450.0, 12345.6, 43200.0, 86000.0])
        expected = np.array([interpolate(nodes, values, t) for t in targets])
        np.testing.assert_allclose(lagrange(nodes, values, targets), expected, rtol=0, atol=1e-9)


class FixabilityTests(unittest.TestCase):
    def test_hindsight_labels(self):
        from gps_processing.processing.ml.fixability import hindsight_samples, FixabilityModel
        rows = []
        rng = np.random.default_rng(0)
        for sat in range(2, 12):
            truth_nw, truth_n1 = sat, -sat
            biased = sat >= 9                                    # converges, but to a fraction
            for k in range(1, 121):
                sigma = 0.5/np.sqrt(k)
                rows.append(dict(epok=30.0*k, sat=sat, reference_sat=1, arc_epochs=k, elevation=40.0,
                                 NW_float=truth_nw+rng.normal(0, sigma/3), sigma_NW=sigma/3, cov_N1_NW=0.0,
                                 N1_float=truth_n1+(0.4 if biased else 0)+rng.normal(0, sigma), sigma_N1=sigma))
        samples = hindsight_samples(pd.DataFrame(rows))
        n1 = samples[samples.stage_n1 == 1]
        self.assertEqual(set(n1[n1.label == 0].sat) >= {9, 10, 11}, True)
        self.assertGreater(n1[n1.sat < 9].label.mean(), 0.5)
        model = FixabilityModel.fit(samples)
        self.assertIsNotNone(model)


class NetworkTests(unittest.TestCase):
    def test_learned_process_noise_and_sparse_fcbs(self):
        from gps_processing.processing.fcb.core import UFOResult, process_noise
        from gps_processing.processing.fcb.network import drop_sparse, network_warnings
        satellite = pd.DataFrame(dict(block=np.tile(np.arange(8), 2), epok=np.tile(np.arange(8)*900.0, 2),
                                      sat=np.repeat([3, 4], 8), ufo=np.r_[np.zeros(8), np.cumsum([0, 0.05, 0.1, -0.08, 0.03, -0.12, 0.07, 0.02])],
                                      sigma=0.01))
        rows = pd.DataFrame(dict(block=[0]*4, sat=[3, 3, 3, 4], station=['A', 'B', 'C', 'A'], fixed=True))
        result = UFOResult('ekf', satellite.copy(), pd.DataFrame(), rows, 'A')
        noise = process_noise(result)
        self.assertGreater(noise[4], noise[3])
        self.assertEqual(drop_sparse(result, 3), 15)              # only (block 0, G03) has three stations
        warnings = network_warnings({'A': SITE, 'B': SITE+1e3}, user_xyz=SITE+3e6)
        self.assertEqual(len(warnings), 2)


if __name__ == '__main__':
    unittest.main()
