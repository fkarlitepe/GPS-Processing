import argparse
from pathlib import Path
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run, compare, VARIANTS
from gps_processing.processing.estimation.options import ESTIMATORS

def main():
    parser = argparse.ArgumentParser(description='TOGU GPS — modular experimental PPP / PPP-AR (EKF and machine-learning options)')
    parser.add_argument('--sp3-before', default='', help='previous-day SP3 (optional for single-day products)')
    parser.add_argument('--sp3-day', type=Path, required=True)
    parser.add_argument('--sp3-after', default='', help='next-day SP3 (optional)')
    parser.add_argument('--rinex', default='', help='observation file (not needed with --fcb-stations)')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--elevation-mask', type=float, default=10)
    parser.add_argument('--technique', choices=['PPP','PPP-AR/SDBS','PPP-AR/DCM','PPP-AR/IPC'], default='PPP')
    parser.add_argument('--ar-products', default='', help='normalized JSON product or Bias-SINEX code+phase OSB file (*.BIA)')
    parser.add_argument('--clock-file', default='', help='RINEX clock file (30 s / 5 s) replacing the SP3 clocks')
    parser.add_argument('--antex', default='', help='ANTEX file: satellite PCO, receiver ARP/PCO/PCV')
    parser.add_argument('--no-phase-windup', dest='phase_windup', action='store_false')
    parser.add_argument('--no-solid-tides', dest='solid_tides', action='store_false', help='skip the IERS 2010 solid Earth tide')
    parser.add_argument('--ionosphere', default='', help='IONEX global ionosphere map in Data/ (second-order IF correction)')
    parser.add_argument('--troposphere', default='', help='SINEX_TRO station ZTD product in Data/ (a-priori ZTD + gradients)')
    parser.add_argument('--code-biases', default='', help='Bias-SINEX DSB/OSB file in Data/: aligns C1C/C2L... codes to C1W/C2W')
    parser.add_argument('--no-clock-jump-repair', dest='repair_clock_jumps', action='store_false',
                        help='keep receiver millisecond clock jumps (default: detected and removed from the phase)')
    parser.add_argument('--ar-ratio', type=float, default=3.0)
    parser.add_argument('--ar-min-epochs', type=int, default=20)
    parser.add_argument('--ar-max-std', type=float, default=0.15)
    estimation = parser.add_argument_group('estimation (EKF)')
    estimation.add_argument('--estimator', choices=ESTIMATORS, default='ekf')
    estimation.add_argument('--dynamics', choices=['static','kinematic'], default='static')
    estimation.add_argument('--position-noise', type=float, default=10.0, help='kinematic random walk, m/√s')
    estimation.add_argument('--troposphere-noise', type=float, default=5.0, help='ZWD random walk, mm/√h')
    estimation.add_argument('--no-troposphere', dest='estimate_troposphere', action='store_false')
    estimation.add_argument('--mapping', dest='tropo_mapping', choices=['niell', 'chao'], default='niell',
                            help='troposphere mapping function (Saastamoinen ZHD a priori, residual ZWD estimated)')
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
    fcb = parser.add_argument_group('FCB estimation from a reference network (Li & Zhang 2012)')
    fcb.add_argument('--fcb-stations', nargs='+', default=[], metavar='RINEX',
                     help='at least 5 reference-station RINEX files of the same day in Data/ (e.g. Data/network/*.23o): '
                          'FCB estimation alone, or the AR product of a PPP-AR run without --ar-products')
    fcb.add_argument('--fcb-method', choices=['batch', 'ekf', 'ekf+ml', 'all'], default='all')
    fcb.add_argument('--fcb-interval', type=float, default=900.0, help='narrow-lane block length, s')
    # With --rinex and a PPP-AR technique but no --ar-products, the FCBs are estimated first and used.
    parser.add_argument('--compare', nargs='+', choices=list(VARIANTS), metavar='METHOD',
                        help='Compare methods on the same data: ' + ' | '.join(VARIANTS))
    args = parser.parse_args()
    names = args.__dict__.pop('compare')
    stations = args.__dict__.pop('fcb_stations')
    args.fcb_files = list(stations)
    if args.reference_xyz is not None:
        args.reference_xyz = tuple(args.reference_xyz)
    progress = lambda n,m: print(f'[{n:3d}%] {m}')
    try:
        if stations and (not args.rinex or args.technique == 'PPP'):
            from gps_processing.processing.fcb.network import estimate_fcb
            from gps_processing.processing.fcb.core import METHODS
            config = ProcessingConfig(**{**vars(args), 'rinex': stations[0]})
            config.validate()
            out = estimate_fcb(config, stations, METHODS if args.fcb_method == 'all' else (args.fcb_method,), args.fcb_interval, progress=progress)
            for method, entry in out['summary']['methods'].items():
                print(method, {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()})
            print('best method (leave-one-station-out):', out['summary']['best_method'], '·', out['output'])
            return out
        if not args.rinex:
            parser.error('--rinex is required (or --fcb-stations for FCB estimation)')
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
