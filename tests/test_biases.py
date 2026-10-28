"""Code-bias alignment (Bias-SINEX) and receiver clock-jump repair (Banville et al. 2008)."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from gps_processing.constants import C, F1, F2
from gps_processing.io.bias_sinex import CodeBiases, align_code_biases, read_bias_sinex
from gps_processing.processing.ar.solver import solve
from gps_processing.processing.clock_jump import MILLISECOND_M, repair_clock_jumps
from gps_processing.processing.combinations import ionosphere_free, melbourne_wubbena
from tests import test_ar

DAY = datetime(2023, 11, 23)


def bias_text(rows):
    """rows: (kind, prn, obs1, obs2, ns)."""
    lines = ['%=BIA 1.00 COD 2023:330:00000 COD 2023:327:00000 2023:328:00000 A 00000034',
             '+BIAS/SOLUTION',
             '*BIAS SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___']
    for kind, prn, obs1, obs2, value in rows:
        lines.append(f' {kind:<4} G063 G{prn:02d}           {obs1:<4} {obs2:<4} 2023:327:00000 2023:328:00000 ns   {value:21.4f} {0.005:11.4f}')
    lines.append(' DSB  G063 G01 MADR00ESP C1C  C1W  2023:327:00000 2023:328:00000 ns   9.0 0.1')   # receiver: ignored
    return '\n'.join(lines+['-BIAS/SOLUTION', '%=ENDBIA'])+'\n'


class CodeBiasTests(unittest.TestCase):
    def read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.BSX'
            path.write_text(bias_text(rows))
            return CodeBiases(read_bias_sinex(path, DAY))

    def test_dsb_reverse_and_osb_pairs(self):
        ns = C*1e-9
        biases = self.read([('DSB', 1, 'C1C', 'C1W', 2.0), ('DSB', 2, 'C1W', 'C1C', 1.0),
                            ('OSB', 3, 'C2L', '', 4.0), ('OSB', 3, 'C2W', '', 1.5)])
        self.assertAlmostEqual(biases.dsb(1, 'C1C', 100), 2.0*ns)
        self.assertAlmostEqual(biases.dsb(2, 'C1C', 100), -1.0*ns)
        self.assertAlmostEqual(biases.dsb(3, 'C2L', 100), 2.5*ns)
        self.assertEqual(biases.dsb(5, 'C1W', 100), 0.0)
        self.assertIsNone(biases.dsb(5, 'C1C', 100))
        self.assertEqual(len(biases.biases), 4)                   # receiver entry skipped

    def test_alignment_reproduces_w_combinations(self):
        ns = C*1e-9
        rho, l1, l2 = 2.2e7, 2.2e7+0.3, 2.2e7-0.7
        dsb1, dsb2 = 2.0*ns, -1.2*ns
        c1c, c2l = rho+dsb1, rho+dsb2                              # observed = W code + DSB
        import pandas as pd
        frame = pd.DataFrame([dict(sat=1, epok=0.0, signals='C1C L1C C2L L2L', Pc=ionosphere_free(c1c, c2l),
                                   Lc=ionosphere_free(l1, l2), A4=melbourne_wubbena(c1c, c2l, l1, l2), travel_code=c1c),
                              dict(sat=2, epok=0.0, signals='C1C L1C C2W L2W', Pc=0.0, Lc=0.0, A4=0.0, travel_code=0.0)])
        out = align_code_biases(frame, self.read([('DSB', 1, 'C1C', 'C1W', 2.0), ('DSB', 1, 'C2L', 'C2W', -1.2)]))
        self.assertAlmostEqual(out.Pc[0], ionosphere_free(rho, rho), places=6)
        self.assertAlmostEqual(out.A4[0], melbourne_wubbena(rho, rho, l1, l2), places=6)
        self.assertAlmostEqual(out.travel_code[0], rho, places=6)
        self.assertEqual(out.signals[0], 'C1W L1C C2W L2L')   # phase label kept
        self.assertTrue(out.code_bias_aligned[0])
        self.assertFalse(out.code_bias_aligned[1])                 # missing DSB: left unaligned
        self.assertEqual(out.signals[1], 'C1C L1C C2W L2W')

    def test_ar_product_accepts_aligned_codes_only(self):
        import json
        import pandas as pd
        from gps_processing.io.ar_products import ARProducts
        product = dict(schema='togu-ar-1', method='DCM', date='2023-11-23', signals=['C1W', 'L1W', 'C2W', 'L2W'],
                       source='synthetic', orbit_product='x.sp3', convention='add_to_observation_m_positive_ambiguity',
                       records=[dict(epok=0, sat=1, code_m=0, phase_m=0, mw_m=0), dict(epok=0, sat=2, code_m=0, phase_m=0, mw_m=0)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'p.json'; path.write_text(json.dumps(product))
            reader = ARProducts(path, 'DCM', DAY)
        obs = pd.DataFrame([dict(epok=0, sat=1, signals='C1W L1C C2W L2W', code_bias_aligned=True),
                            dict(epok=0, sat=2, signals='C1W L1W C2W L2W', code_bias_aligned=False)])
        self.assertEqual(len(reader.apply(obs)), 2)
        with self.assertRaises(ValueError):
            reader.apply(obs.assign(signals='C1C L1C C2W L2W', code_bias_aligned=False))
        with self.assertRaises(ValueError):
            reader.apply(obs.drop(columns='code_bias_aligned').assign(signals='C1W L1C C2W L2W'))


class ClockJumpTests(unittest.TestCase):
    def jumped(self, epoch=450, k=1):
        receiver, obs, sat = test_ar.ARTests().synthetic()
        later = obs.epok >= epoch
        # Code-only receiver clock steering: Pc and the NL code jump, phase is continuous.
        obs.loc[later, 'Pc'] += k*MILLISECOND_M
        obs.loc[later, 'A4'] -= k*MILLISECOND_M
        return receiver, obs, sat

    def test_detects_and_repairs_exact_millisecond(self):
        receiver, clean, _ = test_ar.ARTests().synthetic()
        _, obs, _ = self.jumped(k=-2)
        repaired, jumps = repair_clock_jumps(obs)
        self.assertEqual(jumps, [dict(epok=450.0, jump_ms=-2)])
        np.testing.assert_allclose(repaired.Pc-repaired.Lc, clean.Pc-clean.Lc, atol=1e-6)
        np.testing.assert_allclose(repaired.A4, clean.A4, atol=1e-6)
        _, none = repair_clock_jumps(clean)
        self.assertEqual(none, [])

    def test_ar_keeps_fixes_through_a_clock_jump(self):
        receiver, obs, sat = self.jumped()
        config = SimpleNamespace(technique='PPP-AR/DCM', elevation_mask=10, ar_min_epochs=3, ar_ratio=3, ar_max_std=100.)
        broken, *_ = solve(obs, sat, receiver, config)
        repaired, _ = repair_clock_jumps(obs)
        kept, *_ = solve(repaired, sat, receiver, config)
        after = lambda p: p[(p.epok >= 450) & (p.epok < 450+2*30)]   # ar_min_epochs-1 epochs
        self.assertTrue(after(broken).status.eq('float').all())         # every arc restarted
        self.assertFalse(after(kept).status.eq('float').any())
        np.testing.assert_allclose(kept.iloc[-1][['x', 'y', 'z']].to_numpy(float), receiver, atol=0.02)


if __name__ == '__main__':
    unittest.main()
