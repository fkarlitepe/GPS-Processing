"""FCB/UFO estimation: estimators on a synthetic network and the full workflow round trip."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pandas as pd
from gps_processing.io.fcb_products import FCBProducts
from gps_processing.processing.ar.models import BETA, LAMBDA_N, LAMBDA_W
from gps_processing.processing.fcb import core, network
from tests.fcb_simulation import network as simulate, satellite_error

DAY = datetime(2023, 11, 23)


class EstimatorTests(unittest.TestCase):
    def test_noise_free_network_is_recovered_by_every_method(self):
        exact, truth = simulate(stations=6, sats=10, blocks=12, outliers=0.0, seed=3)
        for method in core.METHODS:
            result = core.estimate(exact, method)
            self.assertLess(satellite_error(result, truth), 0.06, method)
            self.assertGreater(result.rows.fixed.mean(), 0.9, method)

    def test_ekf_beats_independent_blocks_in_leave_one_out(self):
        table, truth = simulate(seed=0)
        loo = {m: core.leave_one_out(table, m).rms_cycles.mean() for m in ('batch', 'ekf')}
        self.assertLess(loo['ekf'], loo['batch'])
        ekf, batch = core.estimate(table, 'ekf'), core.estimate(table, 'batch')
        self.assertLess(satellite_error(ekf, truth), satellite_error(batch, truth))

    def test_ml_learns_low_elevation_scatter(self):
        table, _ = simulate(seed=2)
        rows = core.estimate(table, 'ekf').rows
        sigma = core.learned_sigma(rows)
        low, high = rows.elevation < 20, rows.elevation > 60
        self.assertGreater(np.median(sigma[low]), 1.5*np.median(sigma[high]))
        result = core.estimate(table, 'ekf+ml')
        self.assertEqual(result.method, 'ekf+ml')
        self.assertIn('ml_dropped', result.info)

    def test_wrap_and_circular_mean(self):
        np.testing.assert_allclose(core.wrap([0.6, -0.6, 1.25]), [-0.4, 0.4, 0.25])
        self.assertAlmostEqual(core.circular_mean([0.48, -0.48]), 0.5, places=6)


def station_frames(stations=6, sats=8, hours=3, seed=5, system='G'):
    """Float ambiguities a converged PPP would give: A4 per epoch and the IF ambiguity N3."""
    from gps_processing import gnss
    k = gnss.constants(system)
    LAMBDA_W, LAMBDA_N, BETA = k['lambda_w'], k['lambda_n'], k['beta']
    offset = gnss.SYSTEMS[system]['offset']
    rng = np.random.default_rng(seed)
    s_wl, s_nl = rng.uniform(-0.5, 0.5, sats), rng.uniform(-0.5, 0.5, sats)
    frames = []
    for i in range(stations):
        r_wl, r_nl = rng.uniform(-0.5, 0.5, 2)
        rows = []
        for j in range(sats):
            n_w, n_1 = rng.integers(-10, 10, 2)
            for epoch in np.arange(0, hours*3600, 30.0):
                b_w = n_w+r_wl+s_wl[j]
                rows.append(dict(epok=epoch, sat=offset+j+1, arc=0, elevation=45.0, sigma_N3=0.004,
                                 A4=LAMBDA_W*(b_w+rng.normal(0, 0.15)),
                                 N3=LAMBDA_N*(n_1+r_nl+s_nl[j])+BETA*b_w+rng.normal(0, 0.002)))
        frames.append((f'ST{i:02d}', pd.DataFrame(rows)))
    return frames


class WorkflowTests(unittest.TestCase):
    def test_estimated_fcb_file_makes_user_ambiguities_integer(self):
        frames = station_frames()
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(output=directory, sp3_day='x.sp3', clock_file='', fcb_workers=1)
            calls = iter(frames)
            with mock.patch.object(network, 'validate_stations', return_value=[Path(f'{n}.rnx') for n, _ in frames]), \
                 mock.patch.object(network, 'station_ambiguities', side_effect=lambda c, p: (*(lambda f: (f[0], DAY, f[1], np.zeros(3)))(next(calls)),)):
                out = network.estimate_fcb(config, [], methods=('ekf',), interval=900.0, validate=False)
            products = FCBProducts(Path(directory)/'fcb'/'fcb_ekf.fcb', 'SDBS', DAY)
            self.assertIn('ekf', out['summary']['methods'])
        # User side on station ST00 (same seed): arc-averaged MW, IF ambiguity at 01:00.
        user = station_frames(stations=1, seed=5)[0][1]
        corrected = products.apply(user[user.epok < 3600.0].assign(signals='', dt_sat=0.0))
        corrected = corrected.groupby('sat').agg(A4=('A4', 'mean'), mw_m=('mw_m', 'mean'), N3=('N3', 'last'), phase_m=('phase_m', 'last'))
        wide = (corrected.A4+corrected.mw_m)/LAMBDA_W
        sd_wide = wide.to_numpy()-wide.to_numpy()[0]
        np.testing.assert_allclose(sd_wide, np.rint(sd_wide), atol=0.1)
        exact_wide = np.rint(sd_wide)
        phase = (corrected.N3+corrected.phase_m).to_numpy()
        sd_narrow = (phase-phase[0]-BETA*exact_wide)/LAMBDA_N
        np.testing.assert_allclose(sd_narrow, np.rint(sd_narrow), atol=0.1)

    def test_galileo_fcb_round_trip_uses_e5a_column(self):
        from gps_processing import gnss
        frames = station_frames(seed=7, system='E')
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(output=directory, sp3_day='x.sp3', clock_file='', systems=('G', 'E'), fcb_workers=1)
            calls = iter(frames)
            with mock.patch.object(network, 'validate_stations', return_value=[Path(f'{n}.rnx') for n, _ in frames]), \
                 mock.patch.object(network, 'station_ambiguities', side_effect=lambda c, p: (*(lambda f: (f[0], DAY, f[1], np.zeros(3)))(next(calls)),)):
                out = network.estimate_fcb(config, [], methods=('ekf',), interval=900.0, validate=False)
            self.assertEqual(out['summary']['systems']['G']['stations'], 0)     # no GPS: skipped, not an error
            text = (Path(directory)/'fcb'/'fcb_ekf.fcb').read_text()
            self.assertIn(' E01 ', text)
            products = FCBProducts(Path(directory)/'fcb'/'fcb_ekf.fcb', 'SDBS', DAY)
        k = gnss.constants('E')
        user = station_frames(stations=1, seed=7, system='E')[0][1]
        corrected = products.apply(user[user.epok < 3600.0].assign(signals='', dt_sat=0.0))
        corrected = corrected.groupby('sat').agg(A4=('A4', 'mean'), mw_m=('mw_m', 'mean'), N3=('N3', 'last'), phase_m=('phase_m', 'last'))
        wide = ((corrected.A4+corrected.mw_m)/k['lambda_w']).to_numpy()
        sd_wide = wide-wide[0]
        np.testing.assert_allclose(sd_wide, np.rint(sd_wide), atol=0.1)
        phase = (corrected.N3+corrected.phase_m).to_numpy()
        sd_narrow = (phase-phase[0]-k['beta']*np.rint(sd_wide))/k['lambda_n']
        np.testing.assert_allclose(sd_narrow, np.rint(sd_narrow), atol=0.1)

    def test_l1_l2_conversion_matches_product_reader(self):
        nl, wl = 0.31, -0.22
        b1, b2 = network.l1_l2(nl, wl)
        from gps_processing.constants import C, F1, F2
        if1, if2 = F1**2/(F1**2-F2**2), -F2**2/(F1**2-F2**2)
        self.assertAlmostEqual(-(if1*C/F1*b1+if2*C/F2*b2), -LAMBDA_N*nl, places=12)   # phase_m
        self.assertAlmostEqual(-LAMBDA_W*(b1-b2), -LAMBDA_W*wl, places=12)             # mw_m

    def test_needs_five_stations(self):
        with self.assertRaisesRegex(ValueError, '5'):
            network.validate_stations([])


if __name__ == '__main__':
    unittest.main()
