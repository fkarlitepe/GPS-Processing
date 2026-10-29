"""High-rate clocks, OSB / FCB AR products and antenna models (PCO, PCV, wind-up)."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.constants import C, F1, F2
from gps_processing.data_directory import DATA_DIR
from gps_processing.io.antex import Antenna
from gps_processing.io.clock_rinex import Clocks, read_clock_rinex
from gps_processing.io.fcb_products import FCBProducts
from gps_processing.io.osb_products import OSBProducts
from gps_processing.processing import antenna as models

DAY = datetime(2023, 11, 23)
LAMBDA_W, LAMBDA_NL = C/(F1-F2), C/(F1+F2)
IF1, IF2 = F1**2/(F1**2-F2**2), -F2**2/(F1**2-F2**2)


def write(directory, name, text):
    path = Path(directory)/name
    path.write_text(text)
    return path


class ClockTests(unittest.TestCase):
    def test_read_and_interpolate(self):
        text = ('     3.00           C                                       RINEX VERSION / TYPE\n'
                'GFZ  GeoForschungsZentrum Potsdam                           ANALYSIS CENTER\n'
                '                                                            END OF HEADER\n'
                'AR mgue 2023 11 23  0  0  0.000000  1    0.1E-03\n'
                'AS G05  2023 11 23  0  0  0.000000  1   -0.151401051936E-03\n'
                'AS G05  2023 11 23  0  0 30.000000  1   -0.151401335153E-03\n'
                'AS G05  2023 11 23  0  5  0.000000  1   -0.151400000000E-03\n')
        with tempfile.TemporaryDirectory() as directory:
            frame = read_clock_rinex(write(directory, 'x.CLK', text), DAY)
        self.assertEqual(frame.attrs['agency'], 'GFZ')
        self.assertEqual(len(frame), 3)
        clocks = Clocks(frame)
        self.assertAlmostEqual(clocks(5, 15.0), (-151.401051936-151.401335153)/2, places=9)
        self.assertTrue(clocks.covers(5, 30.0))
        self.assertFalse(clocks.covers(5, 100.0))           # 270 s gap > 120 s
        self.assertFalse(clocks.covers(6, 0.0))


def bias_file(rows):
    lines = ['%=BIA 1.00 GBM 2023:333:49359 GBM 2023:327:00000 2023:328:00000 P 00000 0', '+BIAS/SOLUTION']
    for prn, obs, start, end, ns in rows:
        lines.append(f' OSB       G{prn:02d}           {obs:<4}      2023:327:{start:05d} 2023:327:{end:05d} ns   {ns:21.4f}')
    lines.append(' OSB       G05  MADR00ESP C1W       2023:327:00000 2023:327:00030 ns   9.0')   # station: ignored
    return '\n'.join(lines+['-BIAS/SOLUTION', '%=ENDBIA'])+'\n'


class OSBTests(unittest.TestCase):
    def test_corrections_band_fallback_and_drop(self):
        ns = C*1e-9
        b = dict(C1W=10.0, C2W=12.0, L1C=0.3, L2W=0.5)
        rows = [(5, o, s, s+30, v+(0.01 if o[0] == 'L' and s else 0.0)) for o, v in b.items() for s in (0, 30)]
        with tempfile.TemporaryDirectory() as directory:
            products = OSBProducts(write(directory, 'GBM0MGXRAP_x.BIA', bias_file(rows)), 'IPC', DAY,
                                   'GBM0MGXRAP_20233270000_01D_30S_CLK.CLK')
            with self.assertRaisesRegex(ValueError, 'different'):
                OSBProducts(write(directory, 'WUM0MGXRAP_x.BIA', bias_file(rows)), 'IPC', DAY, 'GBM0MGXRAP_x_CLK.CLK')
        obs = pd.DataFrame([dict(epok=e, sat=s, signals='C1W L1W C2W L2W', dt_sat=0.0) for e in (0.0, 30.0, 60.0) for s in (5, 6)])
        out = products.apply(obs)
        self.assertEqual(len(out), 2)                                    # G06 and epoch 60 have no OSB
        self.assertEqual(products.meta['dropped_rows'], 4)
        c1, c2, l1, l2 = (b[k]*ns for k in ('C1W', 'C2W', 'L1C', 'L2W'))  # L1W phase taken from L1C
        first = out.iloc[0]
        self.assertAlmostEqual(first.code_m, -(IF1*c1+IF2*c2), places=9)
        self.assertAlmostEqual(first.phase_m, -(IF1*l1+IF2*l2), places=9)
        self.assertAlmostEqual(first.mw_m, -((F1*l1-F2*l2)/(F1-F2)-(F1*c1+F2*c2)/(F1+F2)), places=9)
        self.assertNotAlmostEqual(out.iloc[1].phase_m, first.phase_m, places=6)   # 30 s time-variable phase bias


class FCBTests(unittest.TestCase):
    def test_wide_lane_becomes_integer(self):
        text = ('## FRACTIONAL CYCLE BIAS BY GENFCB v.1.6\n#s 3 L1 L2 L5\n#\n'
                '2023/11/23 00:00:00 2023/11/23 00:05:00  G05  -1.85653 0.08586  -2.22089 0.10825   0.00000 0.00000\n'
                '2023/11/23 00:00:00 2023/11/23 00:05:00  E02  -0.62041 0.08792   0.00000 0.00000  -0.62104 0.11475\n')
        with tempfile.TemporaryDirectory() as directory:
            products = FCBProducts(write(directory, 'x.fcb', text), 'SDBS', DAY)
        b1, b2 = -1.85653, -2.22089
        obs = pd.DataFrame([dict(epok=0.0, sat=5, A4=LAMBDA_W*(7+b1-b2), dt_sat=0.0), dict(epok=400.0, sat=5, A4=0.0, dt_sat=0.0)])
        out = products.apply(obs)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual((out.A4+out.mw_m)[0]/LAMBDA_W, 7.0, places=9)
        self.assertAlmostEqual(out.phase_m[0], -(IF1*C/F1*b1+IF2*C/F2*b2), places=9)


class AntennaTests(unittest.TestCase):
    def test_sun_and_nominal_attitude(self):
        sun = models.sun_position([43200.0], DAY)[0]
        declination = np.degrees(np.arcsin(sun[2]/np.linalg.norm(sun)))
        self.assertAlmostEqual(declination, -20.3, delta=0.3)            # 23 November
        self.assertAlmostEqual(np.linalg.norm(sun)/1.495978707e11, 0.987, delta=0.003)
        satellite = np.array([[15e6, 10e6, 18e6]])
        ex, ey, ez = models.nominal_axes(satellite, sun[None, :])
        basis = np.vstack([ex, ey, ez])
        np.testing.assert_allclose(basis@basis.T, np.eye(3), atol=1e-12)
        self.assertLess(np.dot(ez[0], satellite[0]), 0)                  # z points to the Earth
        self.assertAlmostEqual(np.dot(ey[0], sun-satellite[0]), 0.0, delta=1e-3*np.linalg.norm(sun))

    def test_windup_half_cycle_for_reversed_yaw_and_unwrap(self):
        receiver = np.array([4849202.0, -360329.0, 4114913.0])
        satellite = receiver+np.array([[1.2e7, 0.5e7, 1.9e7]])
        los = (receiver-satellite)/np.linalg.norm(receiver-satellite)
        ex, ey, _ = models.nominal_axes(satellite, models.sun_position([0.0], DAY))
        a = models.windup(los, ex, ey, receiver)[0]
        b = models.windup(los, -ex, -ey, receiver)[0]                    # yaw + 180 deg
        self.assertAlmostEqual(abs(((a-b)+0.5) % 1-0.5), 0.5, places=9)
        cycles = models.unwrap(np.array([5, 5, 5]), np.array([0., 30., 60.]), np.array([0.45, -0.48, -0.40]))
        np.testing.assert_allclose(cycles, [0.45, 0.52, 0.60])

    def test_receiver_offset_and_pcv_at_zenith(self):
        receiver = np.array([0.0, 0.0, 6371e3])
        los = -np.array([[0.0, 0.0, 1.0]])                                # satellite straight overhead
        item = Antenna('TEST', '', zenith=np.array([0.0, 90.0]),
                       offset={'G01': np.array([0, 0, 0.09]), 'G02': np.array([0, 0, 0.12])},
                       pcv={'G01': np.array([0.002, 0.0]), 'G02': np.array([0.004, 0.0])})
        ionosphere_free, mw = models.receiver_correction(item, np.array([0.0254, 0, 0]), los, receiver)
        d1, d2 = -(0.0254+0.09)+0.002, -(0.0254+0.12)+0.004
        self.assertAlmostEqual(ionosphere_free[0], IF1*d1+IF2*d2, places=9)
        self.assertAlmostEqual(mw[0], (F1*d1-F2*d2)/(F1-F2)-(F1*d1+F2*d2)/(F1+F2), places=9)


PRODUCTS = DATA_DIR/'fcb'/'post_2023327'


@unittest.skipUnless((PRODUCTS/'GBM0MGXRAP_20233270000_01D_30S_ABS.BIA').is_file(), 'GBM product set not in Data/fcb')
class RealDataTests(unittest.TestCase):
    def test_gbm_osb_wide_lanes_are_integer_up_to_receiver_bias(self):
        from gps_processing import gnss
        from gps_processing.io.rinex import read_rinex
        obs = read_rinex(DATA_DIR/'madr3270.23o').combined
        obs = obs[obs.epok < 720].assign(dt_sat=0.0)   # first 12 min: one arc per satellite (WL is clock free)
        products = OSBProducts(PRODUCTS/'GBM0MGXRAP_20233270000_01D_30S_ABS.BIA', 'IPC', DAY)
        data = products.apply(obs)
        # GPS only: Galileo in this RINEX 2 file has no tracking attribute (C1X/C5X assumed), so its
        # code biases are uncertain; its MW arcs scatter by 0.3-1 cycles and are not tested here.
        for system in ('G',):
            part = data[gnss.systems_of(data.sat.to_numpy()) == system]
            cycles = (part.A4+part.mw_m)/gnss.constants(system)['lambda_w']
            arcs = cycles.groupby(part.sat).agg(['mean', 'std'])
            means = arcs[arcs['std'] < 0.3]['mean'].to_numpy()
            receiver = np.angle(np.exp(2j*np.pi*means).mean())/(2*np.pi)
            residual = np.abs((means-receiver+0.5) % 1-0.5)
            self.assertGreaterEqual(len(means), 4, system)
            self.assertLess(residual.max(), 0.1, (system, residual.round(3)))


if __name__ == '__main__':
    unittest.main()


class TideTests(unittest.TestCase):
    def test_moon_and_tide_magnitudes(self):
        from gps_processing.processing.tides import moon_position, solid_tide
        when = datetime(2023, 11, 27)                                    # full Moon 09:16 UTC
        moon, sun = moon_position([33360.0], when)[0], models.sun_position([33360.0], when)[0]
        elongation = np.degrees(np.arccos(moon@sun/np.linalg.norm(moon)/np.linalg.norm(sun)))
        self.assertGreater(elongation, 174)
        self.assertTrue(356e6 < np.linalg.norm(moon) < 407e6)
        equator = np.array([6378137.0, 0, 0])
        radial = solid_tide(equator, np.arange(0, 86400, 600.0), DAY)@np.array([1.0, 0, 0])
        self.assertTrue(0.2 < radial.max()-radial.min() < 0.6)

    def test_correction_restores_range_to_mean_position(self):
        from gps_processing.processing.tides import apply_solid_tide, solid_tide
        receiver = np.array([4849202.394, -360328.993, 4114913.186])
        satellite = receiver+np.array([1.2e7, 0.4e7, 1.8e7])
        epochs = np.array([0.0, 21600.0])
        moved = receiver+solid_tide(receiver, epochs, DAY)
        ranges = np.linalg.norm(satellite-moved, axis=1)
        combined = pd.DataFrame(dict(epok=epochs, sat=5, Pc=ranges, Lc=ranges))
        satellites = pd.DataFrame(dict(epok=epochs, sat=5, x=satellite[0], y=satellite[1], z=satellite[2]))
        out = apply_solid_tide(combined, satellites, receiver, DAY)
        np.testing.assert_allclose(out.Pc, np.linalg.norm(satellite-receiver), atol=1e-4)


class ProductClockTests(unittest.TestCase):
    def test_osb_and_fcb_corrections_carry_the_satellite_clock(self):
        # The AR solver has no satellite-clock term (JSON contract): code_m/phase_m must include c*dt_sat.
        text = ('## FRACTIONAL CYCLE BIAS BY GENFCB v.1.6\n#s 3 L1 L2 L5\n'
                '2023/11/23 00:00:00 2023/11/23 00:05:00  G05   0.00000 0.08586   0.00000 0.10825   0.00000 0.00000\n')
        with tempfile.TemporaryDirectory() as directory:
            products = FCBProducts(write(directory, 'x.fcb', text), 'SDBS', DAY)
        out = products.apply(pd.DataFrame([dict(epok=0.0, sat=5, A4=0.0, dt_sat=-151.4)]))
        self.assertAlmostEqual(out.code_m[0], C*1e-6*-151.4, places=6)
        self.assertAlmostEqual(out.phase_m[0], C*1e-6*-151.4, places=6)
        with self.assertRaisesRegex(ValueError, 'dt_sat'):
            products.apply(pd.DataFrame([dict(epok=0.0, sat=5, A4=0.0)]))


class ConsistencyTests(unittest.TestCase):
    def test_mixed_centres_and_unverifiable_fcb(self):
        from types import SimpleNamespace
        from gps_processing.pipeline import product_consistency
        base = dict(technique='PPP-AR/IPC', clock_file='GBM0MGXRAP_20233270000_01D_30S_CLK.CLK')
        mixed = SimpleNamespace(**base, ar_products='GBM0MGXRAP_20233270000_01D_30S_ABS.BIA',
                                sp3_day='COD0OPSFIN_20233270000_01D_05M_ORB.SP3')
        self.assertIn('COD', product_consistency(mixed)[0][0])
        same = SimpleNamespace(**base, ar_products='GBM0MGXRAP_20233270000_01D_30S_ABS.BIA',
                               sp3_day='GBM0MGXRAP_20233270000_01D_05M_ORB.SP3')
        self.assertEqual(product_consistency(same), ([], []))
        with tempfile.TemporaryDirectory() as directory:
            own = write(directory, 'own.fcb', '#s 3 L1 L2 L5\n# method ekf; orbit GBM0MGXRAP_20233270000_01D_05M_ORB.SP3; '
                                              'clock GBM0MGXRAP_20233270000_01D_30S_CLK.CLK; stations A\n')
            foreign = write(directory, 'jxf.fcb', '## FRACTIONAL CYCLE BIAS BY GENFCB v.1.6\n#s 3 L1 L2 L5\n')
            self.assertEqual(product_consistency(SimpleNamespace(**{**vars(same), 'ar_products': str(own)})), ([], []))
            other = SimpleNamespace(**{**vars(same), 'ar_products': str(own), 'sp3_day': 'COD0OPSFIN_20233270000_01D_05M_ORB.SP3'})
            self.assertTrue(product_consistency(other)[0])
            self.assertTrue(product_consistency(SimpleNamespace(**{**vars(same), 'ar_products': str(foreign)}))[1])
