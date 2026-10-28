import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
from gps_processing.constants import F1, F2
from gps_processing.io.ionex import read_ionex
from gps_processing.io.sinex_tro import read_sinex_tro, select_site
from gps_processing.processing import ionosphere, ppp_ekf
from gps_processing.processing.atmosphere import Atmosphere
from gps_processing.processing.troposphere import Troposphere
from tests.test_estimation import sky, config, RECEIVER

DAY = datetime(2023, 11, 23)


def ionex_text(maps, lat=(90., -90., -30.), lon=(-180., 180., 60.), exponent=-1):
    """``maps``: [(hour, function(lat, lon) -> TECU)] written in IONEX 1.0 layout."""
    label = lambda text, name: f'{text:<60}{name}'
    lines = [label('     1.0            IONOSPHERE MAPS     GPS', 'IONEX VERSION / TYPE'),
             label('  COSZ', 'MAPPING FUNCTION'), label('  6371.0', 'BASE RADIUS'),
             label(f'  {450.0:6.1f}{450.0:6.1f}{0.0:6.1f}', 'HGT1 / HGT2 / DHGT'),
             label(f'  {lat[0]:6.1f}{lat[1]:6.1f}{lat[2]:6.1f}', 'LAT1 / LAT2 / DLAT'),
             label(f'  {lon[0]:6.1f}{lon[1]:6.1f}{lon[2]:6.1f}', 'LON1 / LON2 / DLON'),
             label(f'{exponent:6d}', 'EXPONENT'), label('', 'END OF HEADER')]
    lats = np.arange(lat[0], lat[1]+lat[2]/2, lat[2]); lons = np.arange(lon[0], lon[1]+lon[2]/2, lon[2])
    for k, (hour, field) in enumerate(maps, 1):
        lines += [label(f'{k:6d}', 'START OF TEC MAP'),
                  label(''.join(f'{v:6d}' for v in (2023, 11, 23, hour, 0, 0)), 'EPOCH OF CURRENT MAP')]
        for b in lats:
            lines.append(label(f'  {b:6.1f}{lon[0]:6.1f}{lon[1]:6.1f}{lon[2]:6.1f}{450.0:6.1f}', 'LAT/LON1/LON2/DLON/H'))
            values = [int(round(field(b, l)/10.0**exponent)) for l in lons]
            for i in range(0, len(values), 16):
                lines.append(''.join(f'{v:5d}' for v in values[i:i+16]))
        lines.append(label(f'{k:6d}', 'END OF TEC MAP'))
    lines.append(label('', 'END OF FILE'))
    return '\n'.join(lines)+'\n'


def sinex_tro_text(site='MADR', ztd_mm=(2400.0, 2410.0), gradients=(0.5, -0.3), xyz=RECEIVER):
    epochs = [f'2023:327:{s:05d}' for s in (0, 3600)]
    rows = '\n'.join(f' {site:<9} {e} {z:7.1f} {1.2:6.1f} {gradients[0]:7.3f} {0.05:6.3f} {gradients[1]:7.3f} {0.05:6.3f}'
                     for e, z in zip(epochs, ztd_mm))
    return (f'%=TRO 2.00 IGS 2023:331:00000 IGS 2023:327:00000 2023:327:86100 P {site}\n'
            '+TROP/DESCRIPTION\n'
            '*_________KEYWORD_____________ __VALUE(S)_______________\n'
            ' TROPO PARAMETER NAMES          TROTOT STDDEV TGNTOT STDDEV TGETOT STDDEV\n'
            ' TROPO PARAMETER UNITS          1e+03  1e+03  1e+03  1e+03  1e+03  1e+03\n'
            '-TROP/DESCRIPTION\n'
            '+TROP/STA_COORDINATES\n'
            '*SITE     PT SOLN T __STA_X_____ __STA_Y_____ __STA_Z_____ SYSTEM REMRK\n'
            f' {site:<9}  A    1 P {xyz[0]:12.3f} {xyz[1]:12.3f} {xyz[2]:12.3f} IGS20  IGS\n'
            '-TROP/STA_COORDINATES\n'
            '+TROP/SOLUTION\n'
            '*SITE     ____EPOCH_____ TROTOT STDDEV  TGNTOT  STDDEV  TGETOT  STDDEV\n'
            f'{rows}\n-TROP/SOLUTION\n%=ENDTRO\n')


class IonosphereTests(unittest.TestCase):
    def test_ionex_reading_and_sun_fixed_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.INX'
            path.write_text(ionex_text([(0, lambda b, l: 20.0), (2, lambda b, l: 40.0)]))
            maps = read_ionex(path, DAY)
        self.assertEqual(maps.tec.shape, (2, 7, 7))
        np.testing.assert_allclose(maps.epochs, [0, 7200])
        self.assertAlmostEqual(float(ionosphere.vtec(maps, 3600, 40.0, -3.0)), 30.0, places=6)
        with self.assertRaises(ValueError):
            ionosphere.vtec(maps, 30*3600, 40.0, -3.0)

    def test_bilinear_is_exact_for_linear_field(self):
        field = lambda b, l: 10+0.2*b+0.05*l
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.INX'
            path.write_text(ionex_text([(0, field)], exponent=-2))
            maps = read_ionex(path, DAY)
        self.assertAlmostEqual(float(ionosphere.bilinear(maps, maps.tec[0], 37.0, 12.0)), field(37.0, 12.0), places=2)

    def test_pierce_point_and_slant_factor(self):
        lat, lon = np.radians(40.0), np.radians(-3.0)
        lat_p, lon_p, factor = ionosphere.pierce_point(lat, lon, np.array([np.pi/2, np.radians(10)]),
                                                       np.array([0.0, 0.0]), 6371.0, 450.0)
        self.assertAlmostEqual(lat_p[0], lat); self.assertAlmostEqual(lon_p[0], lon)
        self.assertAlmostEqual(factor[0], 1.0)
        self.assertGreater(lat_p[1], lat)                        # northward look → northern IPP
        self.assertTrue(2.5 < factor[1] < 3.2)                   # textbook SLM factor at 10°

    def test_dipole_field_magnitude_and_second_order_sign(self):
        equator = ionosphere.dipole_field(np.array([[6371.2e3, 0, 0]]))
        pole = ionosphere.dipole_field(np.array([[0, 0, 6371.2e3]]))
        self.assertTrue(2.5e-5 < np.linalg.norm(equator) < 3.5e-5)
        self.assertAlmostEqual(np.linalg.norm(pole)/np.linalg.norm(equator), 2.0, delta=0.15)
        b = np.array([[0, 0, 4e-5]])
        code, phase = ionosphere.ionosphere_free_second_order(np.array([100.0]), b, np.array([[0, 0, 1.0]]))
        expected = 7527*299792458.0*4e-5*100e16/(F1*F2*(F1+F2))
        self.assertAlmostEqual(-code[0], expected)
        self.assertAlmostEqual(phase[0], -code[0]/2)
        self.assertTrue(0.005 < abs(code[0]) < 0.05)             # cm-level for 100 TECU


class TroposphereProductTests(unittest.TestCase):
    def test_sinex_tro_units_columns_and_site_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'madr.TRO'
            path.write_text(sinex_tro_text(site='MADR00ESP'))
            series = read_sinex_tro(path, DAY)
        item = select_site(series, 'MADR', RECEIVER)
        np.testing.assert_allclose(item.epochs, [0, 3600])
        np.testing.assert_allclose(item.ztd, [2.400, 2.410])
        np.testing.assert_allclose(item.ztd_sigma, 0.0012)
        np.testing.assert_allclose(item.gradient_north, 0.0005); np.testing.assert_allclose(item.gradient_east, -0.0003)
        self.assertIs(select_site(series, 'XXXX', RECEIVER+100), item)           # nearest station
        with self.assertRaises(ValueError):
            select_site(series, 'XXXX', RECEIVER+1e5)

    def test_zenith_slant_equals_product_ztd(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'madr.TRO'
            path.write_text(sinex_tro_text(gradients=(0.0, 0.0)))
            atmosphere = Atmosphere(RECEIVER, DAY, 'MADR', troposphere_file=path)
        up = RECEIVER/np.linalg.norm(RECEIVER)*1e3
        delays = atmosphere.slant(1800, RECEIVER, (RECEIVER+up*2e4)[None, :], np.array([90.0]))
        self.assertAlmostEqual(delays.common[0], 2.405, delta=2e-3)        # Chao mapping ≈ 1 at zenith
        self.assertAlmostEqual(delays.zwd_apriori, 2.405-Troposphere(RECEIVER).zhd, places=6)
        self.assertEqual(atmosphere.initial_zwd, 0.0)


class FilterIntegrationTests(unittest.TestCase):
    def test_external_ztd_fixes_troposphere_without_estimation(self):
        obs, sat = sky(RECEIVER, tropo_zwd=0.15)
        ztd = (Troposphere(RECEIVER).zhd+0.15)*1e3
        start = RECEIVER+np.array([5., -4, 6])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'madr.TRO'
            path.write_text(sinex_tro_text(ztd_mm=(ztd, ztd), gradients=(0.0, 0.0)))
            atmosphere = Atmosphere(RECEIVER, DAY, 'MADR', troposphere_file=path)
        error = lambda out: np.linalg.norm(out.positions.iloc[-1][['x', 'y', 'z']].to_numpy(float)-RECEIVER)
        without = ppp_ekf.solve(obs, sat, start, config(estimate_troposphere=False))
        fixed = ppp_ekf.solve(obs, sat, start, config(estimate_troposphere=False), atmosphere=atmosphere)
        estimated = ppp_ekf.solve(obs, sat, start, config(), atmosphere=atmosphere)
        self.assertGreater(error(without), 0.05)
        self.assertLess(error(fixed), 0.02)
        self.assertAlmostEqual(estimated.positions.zwd_m.iloc[-1], 0.15, delta=0.005)

    def test_gim_second_order_correction_is_applied_consistently(self):
        obs, sat = sky(RECEIVER)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'gim.INX'
            path.write_text(ionex_text([(0, lambda b, l: 120.0), (2, lambda b, l: 120.0)]))
            atmosphere = Atmosphere(RECEIVER, DAY, ionosphere_file=path)
        # Simulate observations that contain the second-order term the model predicts.
        for epoch, frame in sat.groupby('epok'):
            coords = frame[['x', 'y', 'z']].to_numpy()
            elevation = np.degrees(np.arcsin(np.clip(((coords-RECEIVER)@(RECEIVER/np.linalg.norm(RECEIVER)))
                                                     / np.linalg.norm(coords-RECEIVER, axis=1), -1, 1)))
            delays = atmosphere.slant(epoch, RECEIVER, coords, elevation)
            index = obs.index[obs.epok == epoch]
            obs.loc[index, 'Pc'] += delays.code
            obs.loc[index, 'Lc'] += delays.phase
            self.assertTrue(np.all(np.abs(delays.code) > 1e-4) and np.all(np.abs(delays.code) < 0.1))
        start = RECEIVER+np.array([2., -1, 1])
        out = ppp_ekf.solve(obs, sat, start, config(), atmosphere=atmosphere)
        self.assertLess(np.linalg.norm(out.positions.iloc[-1][['x', 'y', 'z']].to_numpy(float)-RECEIVER), 0.02)
        self.assertTrue(np.isfinite(out.positions.stec_tecu).all())
        self.assertGreater(out.positions.iono2_phase_mm.iloc[-1], 0.05)


if __name__ == '__main__':
    unittest.main()
