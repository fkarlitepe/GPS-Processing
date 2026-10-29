import argparse
import sys
from pathlib import Path
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run, compare, VARIANTS
from gps_processing.processing.estimation.options import ESTIMATORS

def main():
    parser = argparse.ArgumentParser(description='TOGU GNSS — modular experimental PPP / PPP-AR for GPS and Galileo (EKF and machine-learning options)')
    parser.add_argument('--sp3-before', default='', help='previous-day SP3 (optional for single-day products)')
    parser.add_argument('--sp3-day', type=Path, required=True)
    parser.add_argument('--sp3-after', default='', help='next-day SP3 (optional)')
    parser.add_argument('--rinex', default='', help='observation file (not needed with --fcb-stations)')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--elevation-mask', type=float, default=10)
    parser.add_argument('--technique', choices=['PPP','PPP-AR/SDBS','PPP-AR/DCM','PPP-AR/IPC'], default='PPP')
    parser.add_argument('--systems', nargs='+', choices=['G', 'E'], default=['G'],
                        help='satellite systems: G = GPS (L1/L2), E = Galileo (E1/E5a; needs multi-GNSS SP3/CLK)')
    parser.add_argument('--ar-products', default='', help='normalized JSON product or Bias-SINEX code+phase OSB file (*.BIA)')
    parser.add_argument('--clock-file', default='', help='RINEX clock file (30 s / 5 s) replacing the SP3 clocks')
    parser.add_argument('--antex', default='', help='ANTEX file: satellite PCO, receiver ARP/PCO/PCV')
    parser.add_argument('--no-phase-windup', dest='phase_windup', action='store_false')
    parser.add_argument('--no-solid-tides', dest='solid_tides', action='store_false', help='skip the IERS 2010 solid Earth tide')
    parser.add_argument('--ocean-loading', default='', help='BLQ ocean tide loading file (Onsala service) for the station')
    parser.add_argument('--erp', default='', help='IGS ERP file (polar motion) for the pole tide')
    parser.add_argument('--no-shapiro', dest='shapiro', action='store_false', help='skip the relativistic (Shapiro) delay')
    parser.add_argument('--attitude', default='', help='ORBEX attitude quaternions (*_ATT.OBX) of the orbit/clock centre')
    parser.add_argument('--no-satellite-pcv', dest='satellite_pcv', action='store_false', help='skip the ANTEX satellite PCV')
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
    ambiguity.add_argument('--ar-validation', choices=['ffrt','ratio','ml','ratio+ml','ffrt+ml'], default='ffrt',
                           help='ffrt: fixed-failure-rate ratio test (default) | ratio: success rate + constant ratio (v2.8)')
    ambiguity.add_argument('--ar-failure-rate', type=float, default=0.001, help='FFRT target failure rate')
    ambiguity.add_argument('--ar-min-ambiguities', type=int, default=4)
    ambiguity.add_argument('--ar-max-exclusions', type=int, default=2, help='satellites dropped (lowest P0) after a rejection')
    ambiguity.add_argument('--ar-min-elevation', type=float, default=15.0, help='deg, AR candidates only')
    ambiguity.add_argument('--ar-min-arc', type=float, default=900.0, help='s, minimum arc duration of an AR candidate')
    ambiguity.add_argument('--ar-hold', action='store_true', help='fix-and-hold: feed repeated fixes back to the filter')
    ambiguity.add_argument('--ar-hold-epochs', type=int, default=10)
    ambiguity.add_argument('--no-ar-candidate-gates', dest='ar_candidate_gates', action='store_false',
                           help='skip the PRIDE fraction/sigma gates of the AR candidates')
    ambiguity.add_argument('--ar-fixability', action='store_true',
                           help='self-supervised ML fixability (hindsight labels from a float pass) orders, gates and triggers AR')
    ambiguity.add_argument('--ar-fixability-threshold', type=float, default=0.5)
    ambiguity.add_argument('--ml-weighting', action='store_true')
    ambiguity.add_argument('--ml-threshold', type=float, default=0.99)
    ambiguity.add_argument('--ml-validator', default='', help='trained validator JSON in Data/')
    fcb = parser.add_argument_group('FCB estimation from a reference network (Li & Zhang 2012)')
    fcb.add_argument('--fcb-stations', nargs='+', default=[], metavar='RINEX',
                     help='at least 5 reference-station RINEX files of the same day in Data/ (e.g. Data/network/*.23o): '
                          'FCB estimation alone, or the AR product of a PPP-AR run without --ar-products')
    fcb.add_argument('--fcb-method', choices=['batch', 'ekf', 'ekf+ml', 'all'], default='all')
    fcb.add_argument('--fcb-interval', type=float, default=900.0, help='narrow-lane block length, s')
    fcb.add_argument('--fcb-free-coordinates', dest='fcb_hold_coordinates', action='store_false',
                     help='do not hold the daily station coordinates in a second float pass')
    fcb.add_argument('--fcb-min-contributors', type=int, default=0, help='stations needed for a satellite FCB of a block (0: stations/10 in [1, 10])')
    fcb.add_argument('--fcb-workers', type=int, default=0, help='parallel station processes (0: CPU count, max 8)')
    # With --rinex and a PPP-AR technique but no --ar-products, the FCBs are estimated first and used.
    parser.add_argument('--compare', nargs='+', choices=list(VARIANTS), metavar='METHOD',
                        help='Compare methods on the same data: ' + ' | '.join(VARIANTS))
    parser.add_argument('--config', default='', help='JSON/TOML/YAML file with ProcessingConfig fields; command-line options override it')
    parser.add_argument('--write-config', default='', metavar='FILE', help='write the resulting configuration to FILE (JSON) and exit')
    parser.add_argument('--diagnose-nl', action='store_true',
                        help='narrow-lane diagnostic: hold the coordinates and report how close the float N1 ambiguities are to integers')
    if '--config' in sys.argv[1:]:
        # Values of the file become the defaults, so that explicit options still win.
        from gps_processing.models import read_config_values
        values = read_config_values(sys.argv[sys.argv.index('--config')+1])
        if 'fcb_files' in values:
            values['fcb_stations'] = values.pop('fcb_files')
        if isinstance(values.get('systems'), tuple):
            values['systems'] = list(values['systems'])
        parser.set_defaults(**values)
        for action in parser._actions:
            if action.dest in values:
                action.required = False
    args = parser.parse_args()
    for key in ('config', 'write_config', 'diagnose_nl'):
        setattr(main, key, args.__dict__.pop(key))
    names = args.__dict__.pop('compare')
    stations = args.__dict__.pop('fcb_stations')
    args.fcb_files = list(stations)
    args.systems = tuple(args.systems)
    if args.reference_xyz is not None:
        args.reference_xyz = tuple(args.reference_xyz)
    progress = lambda n,m: print(f'[{n:3d}%] {m}')
    if main.write_config:
        config = ProcessingConfig(**vars(args))
        config.to_file(main.write_config)
        print(f'Configuration written to {main.write_config}')
        return config
    try:
        if main.diagnose_nl:
            from gps_processing.processing.diagnostics import run_nl_diagnostic
            import json
            report = run_nl_diagnostic(ProcessingConfig(**vars(args)), progress=progress)
            for name, entry in report['variants'].items():
                print(name, json.dumps({k: entry.get(k) for k in ('samples', 'median_abs_fraction', 'within_010', 'within_015', 'verdict')}))
            return report
        if stations and (not args.rinex or args.technique == 'PPP'):
            from gps_processing.processing.fcb.network import estimate_fcb
            from gps_processing.processing.fcb.core import METHODS
            from gps_processing.pipeline import run_directory
            config = ProcessingConfig(**{**vars(args), 'rinex': stations[0]})
            config.validate()
            config.output = run_directory(config.output)
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
