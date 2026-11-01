"""Experiment B, step 1: FCBs from the 50 IGS stations around MADR with the COD orbits/clocks.

    python3 -m experiments.run_network_fcb [workers]
"""
import json
import sys
from dataclasses import replace
from experiments import common
from gps_processing.processing.fcb.network import estimate_fcb
from gps_processing.processing.fcb.core import METHODS


def main():
    stations = sorted(str(p) for p in (common.DATA/'network').glob('*_MO.rnx'))
    config = replace(common.base_config('COD', 'PPP'), output=str(common.OUT/'network_fcb'), reference_xyz=None,
                     fcb_files=stations, fcb_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    out = estimate_fcb(config, stations, METHODS, 900.0, progress=lambda p, m: print(f'[{p:3d}%] {m}', flush=True))
    print(json.dumps(out['summary'], indent=1, default=float)[:4000])
    print('output', out['output'])


if __name__ == '__main__':
    main()
