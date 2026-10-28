import itertools
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from gps_processing.processing.ar.integer import search,condition
from gps_processing.processing.ar.models import combination_covariance,difference_matrix,LAMBDA_W,LAMBDA_N,BETA
from gps_processing.processing.ar.solver import solve

class ARTests(unittest.TestCase):
    def test_integer_search_matches_brute_force(self):
        a=np.array([1.35,-2.2,.7]); q=np.array([[.2,.1,.02],[.1,.3,.05],[.02,.05,.1]])
        integers,scores=search(a,q)
        inv=np.linalg.inv(q)
        grid=sorted([(float((a-z)@inv@(a-z)),z) for z in map(np.array,itertools.product(range(-4,5),repeat=3))],key=lambda p:p[0])
        np.testing.assert_allclose(scores,[v[0] for v in grid[:2]])
        np.testing.assert_equal(integers,np.array([v[1] for v in grid[:2]]))
    def test_budget_exhaustion_not_false_fix(self):
        with self.assertRaises(RuntimeError): search(np.array([.2,.3]),np.eye(2),max_nodes=1)
    def test_covariance_and_clock_elimination(self):
        q=combination_covariance(.7)
        self.assertGreater(np.linalg.eigvalsh(q).min(),0)
        self.assertNotEqual(q[0,2],0)
        d=difference_matrix(4,1)
        np.testing.assert_allclose(d@np.tile([10,20,30],4),0)
        r=d@np.kron(np.eye(4),q)@d.T
        self.assertNotEqual(r[0,3],0)
    def test_condition_uses_cross_covariance(self):
        state=np.array([1.,2.2]); q=np.array([[1,.3],[.3,.2]])
        fixed,cov=condition(state,q,[1],np.array([2]))
        np.testing.assert_allclose(fixed,[.7,2])
        self.assertLess(cov[0,0],q[0,0])
    def synthetic(self):
        receiver=np.array([6378137.,0,0]); rows=[]; satellites=[]
        for epoch in range(30):
            for sat in range(1,8):
                angle=sat*.8+epoch*.015
                d=np.array([.8+sat*.04,np.sin(angle),np.cos(angle)])
                xyz=receiver+d/np.linalg.norm(d)*22e6
                distance=np.linalg.norm(xyz-receiver)
                n1,nw=sat*3,sat-2
                rows.append(dict(epok=epoch*30,sat=sat,Pc=distance+50,Lc=distance+80+LAMBDA_N*n1+BETA*nw,
                    A4=3+LAMBDA_W*nw,epoch_reset=False,code_m=0.,phase_m=0.,mw_m=0.))
                satellites.append(dict(epok=epoch*30,sat=sat,x=xyz[0],y=xyz[1],z=xyz[2]))
        return receiver,pd.DataFrame(rows),pd.DataFrame(satellites)
    def test_models_recover_same_static_solution_and_datum(self):
        receiver,obs,sat=self.synthetic()
        results=[]
        for method in ('SDBS','DCM'):
            config=SimpleNamespace(technique='PPP-AR/'+method,elevation_mask=10,ar_min_epochs=3,ar_ratio=3,ar_max_std=.5)
            result=solve(obs,sat,receiver+np.array([2.,-3.,4.]),config)
            pos,amb,diag,q,_=result
            np.testing.assert_allclose(pos.iloc[-1][['x','y','z']].to_numpy(dtype=float),receiver,atol=.15)
            ref=int(pos.iloc[-1].reference_sat)
            datum=amb[(amb.sat==ref)]
            np.testing.assert_allclose(datum[['N1_float','NW_float']],0)
            self.assertTrue(np.isfinite(q).all())
            results.append(pos.iloc[-1][['x','y','z']].to_numpy(dtype=float))
        np.testing.assert_allclose(*results,atol=.02)
    def test_missing_product_rejected(self):
        from gps_processing.models import ProcessingConfig
        config=ProcessingConfig('a','b','c','d',technique='PPP-AR/SDBS')
        with self.assertRaisesRegex(ValueError,'JSON'): config.validate()

    def test_fixed_branch_and_cycle_slip_reset(self):
        receiver,obs,sat=self.synthetic()
        # Deliberately relaxed precision gate tests the fixed branch on noiseless data;
        # this is not a recommended real-data threshold.
        for method in ('SDBS','DCM'):
            config=SimpleNamespace(technique='PPP-AR/'+method,elevation_mask=10,ar_min_epochs=3,ar_ratio=3,ar_max_std=100.)
            pos,amb,diag,*_=solve(obs,sat,receiver,config)
            self.assertIn('partial_fixed',set(pos.status))
            fixed=amb.dropna(subset=['N1_fixed'])
            np.testing.assert_allclose(fixed.N1_fixed,3*(fixed.sat-fixed.reference_sat))
            np.testing.assert_allclose(fixed.NW_fixed,fixed.sat-fixed.reference_sat)
            reference=int(pos.iloc[-1].reference_sat)
            slipped=obs.copy()
            slipped.loc[(slipped.epok==870)&(slipped.sat==reference),'epoch_reset']=True
            p,_,d,*_=solve(slipped,sat,receiver,config)
            self.assertEqual(p.iloc[-1].status,'float')
            self.assertIn(str(reference),d.iloc[-1].reset_satellites)

    def test_product_contract(self):
        import tempfile,json
        from pathlib import Path
        from datetime import datetime
        from gps_processing.io.ar_products import ARProducts
        product=dict(schema='togu-ar-1',method='DCM',date='2023-11-23',
            signals=['C1W','L1W','C2W','L2W'],source='synthetic test',orbit_product='test.sp3',
            convention='add_to_observation_m_positive_ambiguity',
            records=[dict(epok=0,sat=1,code_m=1,phase_m=2,mw_m=3)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'product.json';path.write_text(json.dumps(product))
            reader=ARProducts(path,'DCM',datetime(2023,11,23))
            obs=pd.DataFrame([dict(epok=0,sat=1,signals='C1W L1W C2W L2W')])
            self.assertEqual(reader.apply(obs).iloc[0].phase_m,2)
            with self.assertRaises(ValueError): reader.apply(obs.assign(epok=30))
            with self.assertRaises(ValueError): reader.apply(obs.assign(signals='C1C L1C C2W L2W'))
            with self.assertRaises(ValueError): ARProducts(path,'SDBS',datetime(2023,11,23))
            with self.assertRaises(ValueError): ARProducts(path,'DCM',datetime(2023,11,24))
