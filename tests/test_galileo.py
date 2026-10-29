"""GPS + Galileo: identifiers, combinations, readers and the multi-system float/AR filters."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from gps_processing import gnss
from gps_processing.constants import C, F1, F2, F5
from gps_processing.processing.ar.solver import solve


class IdentifierTests(unittest.TestCase):
    def test_ids_slots_and_constants(self):
        self.assertEqual(gnss.parse('E12'), 112)
        self.assertEqual(gnss.label(112), 'E12')
        self.assertEqual([gnss.slot(1), gnss.slot(32), gnss.slot(101), gnss.slot(136)], [0, 31, 32, 67])
        e = gnss.constants('E')
        self.assertAlmostEqual(e['lambda_w'], C/(F1-F5))
        self.assertAlmostEqual(e['lambda_n']*0+e['beta'], C*F5/(F1**2-F5**2))
        f1, f2 = gnss.frequencies([5, 105])
        np.testing.assert_allclose(f2, [F2, F5])
        with self.assertRaises(ValueError):
            gnss.sat_id('E', 40)


def synthetic(epochs=30):
    """GPS 1..7 and Galileo 101..105; each system has its own receiver code/phase/MW biases."""
    receiver = np.array([6378137., 0, 0]); rows, sats = [], []
    bias = {'G': (50.0, 80.0, 3.0), 'E': (57.5, 91.25, -1.7)}          # code, phase, MW receiver terms (m)
    for epoch in range(epochs):
        for sat in list(range(1, 8))+list(range(101, 106)):
            k = gnss.constants(gnss.system_of(sat))
            angle = (sat % 100)*0.8+epoch*0.015+(0.4 if sat > 100 else 0)
            d = np.array([.8+(sat % 100)*.04, np.sin(angle), np.cos(angle)])
            xyz = receiver+d/np.linalg.norm(d)*(23e6 if sat > 100 else 22e6)
            distance = np.linalg.norm(xyz-receiver)
            n1, nw = (sat % 100)*3, (sat % 100)-2
            code, phase, mw = bias[gnss.system_of(sat)]
            rows.append(dict(epok=epoch*30, sat=sat, Pc=distance+code, Lc=distance+phase+k['lambda_n']*n1+k['beta']*nw,
                             A4=mw+k['lambda_w']*nw, epoch_reset=False, code_m=0., phase_m=0., mw_m=0.))
            sats.append(dict(epok=epoch*30, sat=sat, x=xyz[0], y=xyz[1], z=xyz[2]))
    return receiver, pd.DataFrame(rows), pd.DataFrame(sats)


class MultiSystemARTests(unittest.TestCase):
    def test_each_model_fixes_both_systems_against_their_own_reference(self):
        receiver, obs, sat = synthetic()
        for method in ('DCM', 'IPC'):
            config = SimpleNamespace(technique='PPP-AR/'+method, elevation_mask=10, ar_min_epochs=3, ar_ratio=3, ar_max_std=100.)
            pos, amb, diag, *_ = solve(obs, sat, receiver+np.array([1., -1, 1]), config)
            np.testing.assert_allclose(pos.iloc[-1][['x', 'y', 'z']].to_numpy(float), receiver, atol=0.02, err_msg=method)
            self.assertIn(pos.iloc[-1].status, ('fixed', 'partial_fixed'), method)
            fixed = amb.dropna(subset=['N1_fixed'])
            self.assertTrue((fixed.sat > 100).any() and (fixed.sat <= 100).any(), method)
            same_system = (fixed.sat > 100) == (fixed.reference_sat > 100)
            self.assertTrue(same_system.all(), method)
            np.testing.assert_allclose(fixed.N1_fixed, 3*(fixed.sat % 100-fixed.reference_sat % 100), atol=1e-5, err_msg=method)
            np.testing.assert_allclose(fixed.NW_fixed, fixed.sat % 100-fixed.reference_sat % 100, atol=1e-5, err_msg=method)
            last = pos.iloc[-1]
            self.assertEqual((last.gps_satellites, last.galileo_satellites), (7, 5))

    def test_sdbs_differences_inside_each_system(self):
        receiver, obs, sat = synthetic()
        config = SimpleNamespace(technique='PPP-AR/SDBS', elevation_mask=10, ar_min_epochs=3, ar_ratio=3, ar_max_std=100.)
        pos, amb, *_ = solve(obs, sat, receiver+np.array([2., -3., 4.]), config)
        np.testing.assert_allclose(pos.iloc[-1][['x', 'y', 'z']].to_numpy(float), receiver, atol=0.05)
        datum = amb[amb.sat == amb.reference_sat]
        self.assertEqual(set(datum.sat > 100), {False, True})            # one datum per system
        np.testing.assert_allclose(datum[['N1_float', 'NW_float']], 0)


class FloatTests(unittest.TestCase):
    def test_galileo_clock_absorbs_inter_system_bias(self):
        from gps_processing.processing import ppp_ekf
        from tests.test_estimation import config
        receiver, obs, sat = synthetic(40)
        obs = obs.assign(GF=0.0, dt_sat=0.0, snr1=45.0, snr2=40.0, Lc=obs.Pc+0.1*(obs.sat % 100))
        out = ppp_ekf.solve(obs, sat, receiver+np.array([3., -2, 1]),
                            config(estimate_troposphere=False, physical_corrections=False))
        np.testing.assert_allclose(out.positions.iloc[-1][['x', 'y', 'z']].to_numpy(float), receiver, atol=0.05)


if __name__ == '__main__':
    unittest.main()
