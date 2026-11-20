import unittest
from itertools import combinations
import numpy as np
import pandas as pd
from gps_processing.processing import selection as S


def sky(n=14, seed=3):
    rng = np.random.default_rng(seed)
    el = np.radians(rng.uniform(10, 88, n))
    az = rng.uniform(0, 2*np.pi, n)
    e = np.c_[np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)]
    systems = np.array([0, 1]*(n//2))
    return e, el, az, systems


class GeometryTest(unittest.TestCase):
    def test_dops_match_direct_inverse(self):
        e, el, az, systems = sky()
        w = np.sin(el)**2
        geo = S.Geometry(e, systems, w, S.wet_mapping(el))
        g = np.c_[-e, systems == 0, systems == 1, S.wet_mapping(el)]
        q = np.linalg.inv(g.T@np.diag(w)@g)
        d = geo.dops(np.ones(len(e), bool))
        self.assertAlmostEqual(d['pdop'][0], np.sqrt(q[0, 0]+q[1, 1]+q[2, 2]))
        self.assertAlmostEqual(d['hdop'][0], np.sqrt(q[0, 0]+q[1, 1]))
        self.assertAlmostEqual(d['vdop'][0], np.sqrt(q[2, 2]))
        self.assertAlmostEqual(d['gdop'][0], np.sqrt(np.trace(q)))

    def test_absent_system_is_decoupled_and_singular_is_inf(self):
        e, el, az, systems = sky()
        geo = S.Geometry(e, systems, np.ones(len(e)))
        gps = systems == 0
        g = np.c_[-e[gps], np.ones(gps.sum())]
        q = np.linalg.inv(g.T@g)
        self.assertAlmostEqual(geo.dops(gps)['gdop'][0], np.sqrt(np.trace(q)))
        three = np.zeros(len(e), bool)
        three[:3] = True
        self.assertTrue(np.isinf(geo.dops(three)['pdop'][0]))


class AlgorithmTest(unittest.TestCase):
    def context(self, k, seed=3):
        e, el, az, systems = sky(seed=seed)
        geo = S.Geometry(e, systems, np.sin(el)**2, S.wet_mapping(el))
        return S.Context(geo, 'pdop', el, az, e, np.sin(el)**2, k, np.random.default_rng(0))

    def test_every_method_returns_k_feasible_satellites(self):
        for name, algorithm in S.ALGORITHMS.items():
            c = self.context(8)
            sel = algorithm(c)
            self.assertEqual(len(set(sel)), 8, name)
            counts = np.bincount(c.systems[sel], minlength=2)
            self.assertTrue(all(v == 0 or v >= 2 for v in counts), name)
            self.assertTrue(np.isfinite(c.score([sel])[0]), name)

    def test_exhaustive_is_optimal_and_local_search_never_worse(self):
        for seed in range(4):
            c = self.context(6, seed)
            best = min(c.score(list(combinations(range(c.n), 6))))
            self.assertAlmostEqual(c.score([S.select_exhaustive(c)])[0], best)
            for base, refined in (('fibo-gnss', 'fibo-gnss+ls'), ('backward', 'backward+ls')):
                a, b = c.score([S.ALGORITHMS[base](c), S.ALGORITHMS[refined](c)])
                self.assertLessEqual(b, a+1e-12)
                self.assertGreaterEqual(b, best-1e-12)


class WarmStartTest(unittest.TestCase):
    def test_seed_is_a_candidate_and_evaluations_are_counted(self):
        e, el, az, systems = sky(16, seed=5)
        w = np.sin(el)**2
        geo = S.Geometry(e, systems, w)
        best, value = None, np.inf
        for s in combinations(range(16), 6):
            v = geo.score(S.masks_of([s], 16), 'pdop')[0]
            if v < value:
                best, value = list(s), v
        c = S.Context(geo, 'pdop', el, az, e, w, 6, np.random.default_rng(0))
        c.seed = best
        sel = S.select_fibo_gnss(c)
        self.assertAlmostEqual(c.score([sel])[0], value)
        self.assertGreater(c.evaluations, 0)


class SelectorTest(unittest.TestCase):
    def frame(self, n=14, seed=3):
        e, el, az, systems = sky(n, seed)
        receiver = np.array([4849202.4, -360329.0, 4114913.2])
        R = S.enu_rotation(receiver)
        xyz = receiver+2.2e7*(e@R)
        sats = np.where(systems == 0, np.arange(1, n+1), 100+np.arange(1, n+1))
        return pd.DataFrame(dict(sat=sats, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], elevation=np.degrees(el),
                                 cn0=40+10*np.sin(el))), receiver

    def test_selection_and_analysis_tables(self):
        frame, receiver = self.frame()
        sel = S.SatelliteSelector(S.SelectionOptions('fibo-gnss+ls', 8, 'cn0', analysis=True))
        out = sel.select(0.0, frame, receiver)
        self.assertEqual(len(out), 8)
        epochs, sats = sel.tables()
        self.assertEqual(epochs.selected.iloc[0], 8)
        self.assertEqual(int(sats.selected.sum()), 8)
        self.assertGreaterEqual(epochs.wpdop_selected.iloc[0], epochs.wpdop_all.iloc[0])
        # removing a used satellite never lowers the DOP
        self.assertTrue((sats[sats.selected].delta_dop >= -1e-9).all())

    def test_hysteresis_and_interval_keep_the_subset(self):
        frame, receiver = self.frame()
        for options in (S.SelectionOptions('backward', 8, 'none', interval=300, analysis=True),
                        S.SelectionOptions('backward', 8, 'none', hysteresis=0.99, analysis=True)):
            sel = S.SatelliteSelector(options)
            first = sel.select(0.0, frame, receiver).sat.tolist()
            moved = frame.copy()
            moved[['x', 'y', 'z']] = moved[['x', 'y', 'z']].to_numpy()[::-1]   # a very different sky
            self.assertEqual(sel.select(30.0, moved, receiver).sat.tolist(), first)
            self.assertFalse(sel.tables()[0].reselected.iloc[1])

    def test_validation(self):
        with self.assertRaises(ValueError):
            S.SelectionOptions('fibo-gnss', 4).validate()
        with self.assertRaises(ValueError):
            S.SelectionOptions('nearest', 8).validate()
        S.SelectionOptions('all', 0).validate()


class RevisionTwoTest(unittest.TestCase):
    """FIBO-GNSS2, the Woodbury exchange scores, BDS identifiers and the restricted (active-state) update."""

    def context(self, n=18, k=8, systems=3, seed=4):
        rng = np.random.default_rng(seed)
        el = np.arcsin(rng.uniform(np.sin(np.radians(10)), 1, n))
        az = rng.uniform(0, 2*np.pi, n)
        e = np.stack([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)], 1)
        sy = np.arange(n) % systems
        w = rng.uniform(0.2, 2.0, n)
        return S.Context(S.Geometry(e, sy, w, S.wet_mapping(el)), 'pdop', el, az, e, w, k, rng)

    def test_fast_swap_scores_equal_direct_evaluation(self):
        c = self.context()
        start = S.select_backward(c)
        result = c.fast_swap(list(start))
        self.assertLessEqual(c.score([result])[0], c.score([start])[0]+1e-12)
        # no single feasible exchange improves the result (local optimum of the same neighbourhood)
        counts = np.bincount(c.systems[result], minlength=3)
        best = c.score([result])[0]
        for a in result:
            for b in set(range(c.n))-set(result):
                sa, sb = c.systems[a], c.systems[b]
                if sa == sb or (counts[sa] >= 3 and counts[sb] >= 2):
                    self.assertGreaterEqual(c.score([[x for x in result if x != a]+[b]])[0], best-1e-9)

    def test_fibo_gnss2_is_feasible_and_not_worse_than_its_candidates(self):
        for seed in range(5):
            c = self.context(seed=seed)
            sel = S.select_fibo_gnss2(c)
            self.assertEqual(len(set(sel)), c.k)
            counts = np.bincount(c.systems[sel], minlength=3)
            self.assertTrue(all(v == 0 or v >= 2 for v in counts))
            self.assertLessEqual(c.score([sel])[0], c.score([S.select_fibo_gnss(c)])[0]+1e-9)

    def test_bds_identifiers(self):
        from gps_processing import gnss
        sat = gnss.parse('C19')
        self.assertEqual((sat, gnss.system_of(sat), gnss.label(sat), gnss.slot(sat)), (219, 'C', 'C19', 86))
        self.assertEqual(gnss.slots_for(('G', 'E')), 68)
        f1, f2 = gnss.frequencies([219])
        self.assertAlmostEqual(f1[0], 1561.098e6)
        self.assertAlmostEqual(f2[0], 1268.52e6)

    def test_restricted_update_equals_full_update(self):
        from gps_processing.processing.estimation import ExtendedKalmanFilter
        rng = np.random.default_rng(2)
        dim, active = 9, np.array([0, 1, 2, 4, 6])
        P = np.eye(dim)*4.0
        A = rng.normal(size=(5, 5))
        P[np.ix_(active, active)] = A@A.T+np.eye(5)
        H = np.zeros((6, dim))
        H[:, active] = rng.normal(size=(6, 5))
        y = rng.normal(size=6)
        model = lambda x, inflation: (y, H@x, H, np.eye(6)*0.5, np.arange(6))
        full = ExtendedKalmanFilter(np.zeros(dim), P.copy())
        part = ExtendedKalmanFilter(np.zeros(dim), P.copy())
        full.update(model, 6)
        report = part.update(model, 6, active=active)
        np.testing.assert_allclose(part.x, full.x, atol=1e-10)
        np.testing.assert_allclose(part.P, full.P, atol=1e-10)
        self.assertEqual(report.gain.shape, (dim, 6))

    def test_drop_state_option(self):
        S.SelectionOptions('fibo-gnss2', 8, 'filter', state='drop').validate()
        with self.assertRaises(ValueError):
            S.SelectionOptions('fibo-gnss2', 8, 'filter', state='forget').validate()


if __name__ == '__main__':
    unittest.main()
