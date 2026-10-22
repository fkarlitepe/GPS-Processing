import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.constants import C, F1, F2
from gps_processing.processing.combinations import ionosphere_free, melbourne_wubbena
from gps_processing.processing.orbit import interpolate
from gps_processing.processing.geometry import elevation
from gps_processing.processing.ppp import solve
from gps_processing.io.rinex import read_rinex
from gps_processing.io.sp3 import combine_sp3, read_sp3
from gps_processing.models import ProcessingConfig

ROOT = Path(__file__).resolve().parents[1]

class ProcessingTests(unittest.TestCase):
    def test_combinations_cancel_ionosphere(self):
        distance, ion = 22e6, 8.0
        ratio = F1**2/F2**2
        self.assertAlmostEqual(ionosphere_free(distance+ion,distance+ratio*ion),distance)
        self.assertAlmostEqual(ionosphere_free(distance-ion,distance-ratio*ion),distance)
        self.assertAlmostEqual(melbourne_wubbena(distance,distance,distance,distance),0,places=7)

    def test_interpolation_polynomial_and_boundaries(self):
        nodes = np.arange(-5,7)*900.0
        values = np.c_[nodes/900, (nodes/900)**2, (nodes/900)**3]
        np.testing.assert_allclose(interpolate(nodes,values,225),[.25,.0625,.015625],atol=1e-10)
        np.testing.assert_allclose(interpolate(nodes,values,nodes[0]),values[0])
        with self.assertRaises(ValueError): interpolate(nodes,values,1e8)

    def test_elevation(self):
        np.testing.assert_allclose(elevation(np.array([6378137.,0,0]),np.array([[26378137.,0,0],[6378137.,2e7,0]])),[90,0])

    def test_synthetic_position_clock_and_ambiguities(self):
        receiver=np.array([6378137.,0,0])
        directions=np.array([[1,.2,.2],[1,-.8,.2],[1,.2,-.8],[1,.8,.8],[1,-.6,-.8],[1,.8,-.4]])
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        xyz=receiver+directions*22e6
        observations, satellites=[],[]
        for epoch in (0,30,60):
            for k,coord in enumerate(xyz,1):
                distance=np.linalg.norm(coord-receiver)
                observations.append((epoch,k,distance+120,distance+120+k*2,0.,False,0.))
                satellites.append((epoch,k,*coord))
        obs=pd.DataFrame(observations,columns=['epok','sat','Pc','Lc','A4','epoch_reset','dt_sat'])
        sat=pd.DataFrame(satellites,columns=['epok','sat','x','y','z'])
        positions, amb, diag, covariance, _=solve(obs,sat,receiver+np.array([10,-20,15]))
        np.testing.assert_allclose(positions.iloc[-1][['x','y','z']].to_numpy(dtype=float),receiver,atol=.2)
        self.assertAlmostEqual(positions.iloc[-1].receiver_clock_s*C,120,delta=.2)
        self.assertTrue(np.isfinite(covariance).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(),-1e-7)
        self.assertEqual(len(amb),18)

    def test_sample_readers(self):
        obs=read_rinex(next(ROOT.glob('*.rnx')))
        self.assertEqual(obs.date.strftime('%Y-%m-%d'),'2023-11-23')
        self.assertEqual(len(obs.approximate_xyz),3)
        self.assertTrue(np.isfinite(obs.combined[['Pc','Lc','A4']]).all().all())
        self.assertTrue({'c1c','l1c','c1w','l1w','c2x','l2x','c2w','l2w','c5x','l5x','c1x','l1x'} <= set(obs.raw))
        paths=sorted(ROOT.glob('*.SP3'))
        orbit=combine_sp3(*paths)
        self.assertLess(orbit.epok.min(),0)
        self.assertGreater(orbit.epok.max(),86400)
        with self.assertRaises(ValueError): combine_sp3(paths[1],paths[0],paths[2])

    def test_sp3_variable_satellite_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.sp3'
            path.write_text('*  2023 11 23 00 00 00.00000000\n'+f'PG01{10000.:14.6f}{20000.:14.6f}{21000.:14.6f}{1.:14.6f}\n'+'*  2023 11 23 00 15 00.00000000\n'+f'PG02{11000.:14.6f}{20000.:14.6f}{21000.:14.6f}{2.:14.6f}\n')
            frame=read_sp3(path)
            self.assertEqual(frame.sat.tolist(),[1,2])
            self.assertEqual((frame.timestamp.iloc[1]-frame.timestamp.iloc[0]).total_seconds(),900)

    def test_unsupported_mode_explicit(self):
        config=ProcessingConfig(Path('a'),Path('b'),Path('c'),Path('d'),technique='PPP-AR/SDBS')
        with self.assertRaises(NotImplementedError): config.validate()

if __name__ == '__main__': unittest.main()
