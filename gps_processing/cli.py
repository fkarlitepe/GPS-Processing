import argparse
from pathlib import Path
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run

def main():
    parser = argparse.ArgumentParser(description='TOGU GPS — modüler deneysel float PPP')
    parser.add_argument('--sp3-before', type=Path, required=True)
    parser.add_argument('--sp3-day', type=Path, required=True)
    parser.add_argument('--sp3-after', type=Path, required=True)
    parser.add_argument('--rinex', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--elevation-mask', type=float, default=10)
    args = parser.parse_args()
    try:
        result = run(ProcessingConfig(**vars(args)), lambda n,m: print(f'[{n:3d}%] {m}'))
    except (ValueError, OSError, NotImplementedError) as exc:
        parser.exit(1, f'Hata: {exc}\n')
    print(result.positions.tail().to_string(index=False))
    return result
