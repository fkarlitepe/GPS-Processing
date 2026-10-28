import argparse
from pathlib import Path
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run, compare, VARIANTS
from gps_processing.processing.estimation.options import ESTIMATORS

def main():
    parser = argparse.ArgumentParser(description='TOGU GPS — modular experimental PPP / PPP-AR (EKF and machine-learning options)')
    parser.add_argument('--sp3-before', type=Path, required=True)
    parser.add_argument('--sp3-day', type=Path, required=True)
    parser.add_argument('--sp3-after', type=Path, required=True)
    parser.add_argument('--rinex', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--elevation-mask', type=float, default=10)
    parser.add_argument('--technique', choices=['PPP','PPP-AR/SDBS','PPP-AR/DCM'], default='PPP')
    parser.add_argument('--ar-products', default='')
    parser.add_argument('--ionosphere', default='', help='IONEX global ionosphere map in Data/ (second-order IF correction)')
    parser.add_argument('--troposphere', default='', help='SINEX_TRO station ZTD product in Data/ (a-priori ZTD + gradients)')
    parser.add_argument('--ar-ratio', type=float, default=3.0)
    parser.add_argument('--ar-min-epochs', type=int, default=20)
    parser.add_argument('--ar-max-std', type=float, default=0.15)
    estimation = parser.add_argument_group('estimation (EKF)')
    estimation.add_argument('--estimator', choices=ESTIMATORS, default='ekf')
    estimation.add_argument('--dynamics', choices=['static','kinematic'], default='static')
    estimation.add_argument('--position-noise', type=float, default=10.0, help='kinematic random walk, m/√s')
    estimation.add_argument('--troposphere-noise', type=float, default=5.0, help='ZWD random walk, mm/√h')
    estimation.add_argument('--no-troposphere', dest='estimate_troposphere', action='store_false')
    estimation.add_argument('--no-physical-corrections', dest='physical_corrections', action='store_false')
    estimation.add_argument('--no-robust', dest='robust', action='store_false')
    estimation.add_argument('--code-sigma', type=float, default=0.3)
    estimation.add_argument('--phase-sigma', type=float, default=0.003)
    estimation.add_argument('--reference-xyz', type=float, nargs=3, metavar=('X','Y','Z'))
    ambiguity = parser.add_argument_group('ambiguity resolution and machine learning')
    ambiguity.add_argument('--integer-method', choices=['lambda','ils','bootstrap','cascade'], default='lambda',
                           help='lambda: ILS+PAR | ils: full ILS (v2.0) | bootstrap: decorrelated IB+PAR | cascade: TCAR/CIR-style rounding')
    ambiguity.add_argument('--ar-success-rate', type=float, default=0.999)
    ambiguity.add_argument('--ar-validation', choices=['ratio','ml','ratio+ml'], default='ratio')
    ambiguity.add_argument('--ml-weighting', action='store_true')
    ambiguity.add_argument('--ml-threshold', type=float, default=0.99)
    ambiguity.add_argument('--ml-validator', default='', help='trained validator JSON in Data/')
    parser.add_argument('--compare', nargs='+', choices=list(VARIANTS), metavar='METHOD',
                        help='Compare methods on the same data: ' + ' | '.join(VARIANTS))
    args = parser.parse_args()
    names = args.__dict__.pop('compare')
    if args.reference_xyz is not None:
        args.reference_xyz = tuple(args.reference_xyz)
    progress = lambda n,m: print(f'[{n:3d}%] {m}')
    try:
        if names:
            table, _ = compare(ProcessingConfig(**vars(args)), names, progress)
            print(table.to_string(index=False))
            return table
        result = run(ProcessingConfig(**vars(args)), progress)
    except (ValueError, OSError, NotImplementedError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(result.positions.tail().to_string(index=False))
    print({k: v for k, v in result.statistics.items() if k != 'solution'})
    return result
