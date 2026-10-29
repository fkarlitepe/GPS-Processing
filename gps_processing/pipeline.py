from dataclasses import replace, asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.io.rinex import read_rinex
from gps_processing.io.sp3 import combine_sp3
from gps_processing.io.export import export_result
from gps_processing.processing.orbit import interpolate_satellites
from gps_processing.processing.ppp import solve
from gps_processing.models import ProcessingResult
from gps_processing import analysis

# Named method variants for side-by-side comparison; AR-only variants need an AR technique.
VARIANTS = {
    'Sequential LS (v2.0)': dict(estimator='sequential', ml_weighting=False, integer_method='ils', ar_validation='ratio'),
    'EKF': dict(estimator='ekf', robust=False, ml_weighting=False),
    'Adaptive + robust EKF': dict(estimator='adaptive-ekf', robust=True, ml_weighting=False),
    'EKF + ML weighting': dict(estimator='ekf', robust=True, ml_weighting=True),
    'AR · ILS + ratio': dict(integer_method='ils', ar_validation='ratio'),
    'AR · LAMBDA + PAR': dict(integer_method='lambda', ar_validation='ratio'),
    'AR · LAMBDA + ML validation': dict(integer_method='lambda', ar_validation='ml'),
    'AR · Bootstrapping (decorrelated)': dict(integer_method='bootstrap', ar_validation='ratio'),
    'AR · Cascade rounding (TCAR/CIR)': dict(integer_method='cascade', ar_validation='ratio'),
}
AR_ONLY = {name for name in VARIANTS if name.startswith('AR ·')}


def select_systems(combined, systems, name=''):
    """Keep the satellite systems chosen by the user (GPS 'G', Galileo 'E')."""
    from gps_processing import gnss
    keep = combined[np.isin(gnss.systems_of(combined.sat.to_numpy()), list(systems))]
    missing = [gnss.NAMES[s] for s in systems if not np.isin(gnss.systems_of(keep.sat.to_numpy()), [s]).any()]
    if keep.empty:
        raise ValueError(f'{name}: no dual-frequency observations of the selected systems ({", ".join(gnss.NAMES[s] for s in systems)}).')
    keep = keep.reset_index(drop=True)
    keep.attrs['missing_systems'] = missing
    return keep


def prepare(config, progress):
    progress(5, 'Reading SP3 files')
    orbit = combine_sp3(config.sp3_before, config.sp3_day, config.sp3_after)
    progress(20, 'RINEX and signal combinations')
    obs = read_rinex(config.rinex)
    obs.combined = select_systems(obs.combined, config.systems, Path(config.rinex).name)
    if config.repair_clock_jumps:
        from gps_processing.processing.clock_jump import repair_clock_jumps
        obs.combined, obs.clock_jumps = repair_clock_jumps(obs.combined)
    if config.code_biases:
        from gps_processing.io.bias_sinex import CodeBiases, align_code_biases, read_bias_sinex
        progress(30, 'Aligning code biases to C1W/C2W (Bias-SINEX)')
        obs.combined = align_code_biases(obs.combined, CodeBiases(read_bias_sinex(config.code_biases, obs.date)))
    day = orbit.loc[orbit.epok == 0, 'timestamp']
    if day.empty or day.iloc[0].date() != obs.date.date():
        raise ValueError('RINEX gözlem günü ile ölçüm günü SP3 tarihi eşleşmiyor.')
    from gps_processing import gnss
    for system in config.systems:
        if not np.isin(gnss.systems_of(orbit.sat.to_numpy()), [system]).any():
            raise ValueError(f'{Path(str(config.sp3_day)).name} has no {gnss.NAMES[system]} satellites: '
                             'choose a multi-GNSS orbit (e.g. *MGX*) or deselect that system.')
    clocks = None
    if config.clock_file:
        from gps_processing.io.clock_rinex import Clocks, read_clock_rinex
        progress(35, 'Reading high-rate satellite clocks (RINEX CLK)')
        frame = read_clock_rinex(config.clock_file, obs.date)
        for system in config.systems:
            if not np.isin(gnss.systems_of(frame.sat.to_numpy()), [system]).any():
                raise ValueError(f'{Path(config.clock_file).name} has no {gnss.NAMES[system]} clocks: '
                                 'choose a multi-GNSS clock file or deselect that system.')
        clocks = Clocks(frame)
    progress(40, 'Satellite clocks, 9th-order Lagrange interpolation and relativity')
    obs.combined, satellites = interpolate_satellites(orbit, obs.combined, clocks)
    return orbit, obs, satellites


@lru_cache(maxsize=2)
def load_antex(path, receiver):
    from gps_processing.io.antex import read_antex
    return read_antex(path, receiver=[receiver, receiver[:16].ljust(16)+'NONE'])


def antenna_models(config, obs, satellites):
    """Phase wind-up and solid Earth tide (EKF and AR; the sequential v2.0 baseline stays
    unchanged) and, with an ANTEX file, satellite PCO plus receiver ARP/PCO/PCV. Applied per
    variant on copies."""
    baseline = config.technique == 'PPP' and config.estimator == 'sequential'
    if baseline or not (config.phase_windup or config.antex or config.solid_tides):
        return obs, satellites, {}
    combined, info = obs.combined, {}
    if config.phase_windup or config.antex:
        from gps_processing.processing.antenna import apply_antenna_models
        antennas = load_antex(str(config.antex), obs.antenna_type.ljust(20)[:20]) if config.antex else None
        combined, satellites, info = apply_antenna_models(combined, satellites, obs.approximate_xyz, obs.date,
                                                          antennas, obs.antenna_type, obs.antenna_delta, config.phase_windup)
    if config.solid_tides:
        from gps_processing.processing.tides import apply_solid_tide
        combined = apply_solid_tide(combined, satellites, obs.approximate_xyz, obs.date)
        info['solid_earth_tide'] = 'IERS 2010 degree 2 (tide-free)'
    return replace(obs, combined=combined), satellites, info


def learn_weights(config, obs, satellites, progress, atmosphere=None):
    from gps_processing.processing import ppp_ekf
    from gps_processing.processing.ml.stochastic import ElevationModel, LearnedVarianceModel
    progress(55, 'ML learning pass: stochastic model from float EKF residuals')
    learning = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, config, atmosphere=atmosphere)
    return LearnedVarianceModel.fit(learning.residuals, ElevationModel(config.code_sigma, config.phase_sigma))


def estimate(config, orbit, obs, satellites, progress=None):
    progress = progress or (lambda percent, message: None)
    obs, satellites, models = antenna_models(config, obs, satellites)
    atmosphere = None
    if config.technique != 'PPP' or config.estimator != 'sequential':
        # Built here so the mapping function gets the day of year (Niell annual term).
        from gps_processing.processing.atmosphere import Atmosphere
        if config.ionosphere or config.troposphere:
            progress(50, 'Reading external atmosphere products (IONEX / SINEX_TRO)')
        atmosphere = Atmosphere.from_config(config, obs)
    variance_model = learn_weights(config, obs, satellites, progress, atmosphere) if config.ml_weighting else None
    residuals, products, validator = None, None, None
    progress(65, f'{config.technique} · {config.estimator} estimation')
    if config.technique == 'PPP':
        if config.estimator == 'sequential':
            gps = obs.combined[obs.combined.sat <= 100]              # the v2.0 baseline is GPS only
            positions, ambiguities, diagnostics, covariance, corrections = solve(gps, satellites, obs.approximate_xyz, config.elevation_mask)
        else:
            from gps_processing.processing import ppp_ekf
            output = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, config, variance_model, atmosphere)
            positions, ambiguities, diagnostics, covariance, corrections = output
            residuals = output.residuals
    else:
        from gps_processing.io.ar_products import ARProducts
        from gps_processing.processing.ar.solver import solve as solve_ar
        method = config.technique.split('/')[-1]
        if is_osb(config.ar_products):
            from gps_processing.io.osb_products import OSBProducts
            progress(58, 'Reading code and phase OSBs (Bias-SINEX)')
            products = OSBProducts(config.ar_products, method, obs.date, config.clock_file)
        elif str(config.ar_products).lower().endswith('.fcb'):
            from gps_processing.io.fcb_products import FCBProducts
            products = FCBProducts(config.ar_products, method, obs.date)
        else:
            products = ARProducts(config.ar_products, method, obs.date)
        check_product_systems(products, config.systems, Path(config.ar_products).name)
        if products.meta['schema'] == 'togu-ar-1' and products.meta['orbit_product'] != Path(config.sp3_day).name:
            raise ValueError('AR ürününün orbit_product alanı ölçüm günü SP3 dosyasıyla eşleşmiyor.')
        if 'ml' in config.ar_validation:
            from gps_processing.processing.ml.ar_validation import ARValidator, default_validator
            progress(60, 'Preparing ML ambiguity validator')
            validator = ARValidator.load(config.ml_validator) if config.ml_validator else default_validator()
        combined = products.apply(obs.combined)
        positions, ambiguities, diagnostics, covariance, corrections = solve_ar(
            combined, satellites, obs.approximate_xyz, config, variance_model, validator, atmosphere)
    result = ProcessingResult(positions, ambiguities, diagnostics, covariance, obs, satellites, orbit, corrections, config.technique)
    result.residuals = residuals
    result.settings = settings(config)
    result.settings['clock_jumps'] = obs.clock_jumps
    result.settings.update(models)
    result.settings['orbit_clock_dropped_rows'] = int(obs.combined.attrs.get('orbit_clock_dropped', 0))
    if 'code_bias_aligned' in obs.combined:
        result.settings['code_bias_aligned_rows'] = int(obs.combined.code_bias_aligned.sum())
    if atmosphere is not None:
        result.settings.update(atmosphere.describe())
    if products is not None:
        result.ar_metadata = {k: v for k, v in products.meta.items() if k != 'records'}
        # Which service the user chose (file-name prefix, e.g. JXF, GBM, COD, GRG): kept for comparisons.
        result.ar_metadata['service'] = Path(config.ar_products).name[:3].upper()
        result.ar_metadata['product_file'] = str(config.ar_products)
        result.ar_metadata.update(ratio=config.ar_ratio, min_epochs=config.ar_min_epochs, max_std_cycles=config.ar_max_std,
                                  integer_method=config.integer_method, success_rate=config.ar_success_rate,
                                  validation=config.ar_validation)
    if variance_model is not None:
        result.ml_models['stochastic_model'] = variance_model.to_dict()
    if validator is not None:
        result.ml_models['ar_validator'] = validator.to_dict()
    reference, source = reference_coordinate(config, result)
    result.positions = analysis.add_enu(result.positions, reference)
    result.statistics = analysis.statistics(result.positions, reference, config.technique, source)
    return result


def reference_coordinate(config, result):
    if config.reference_xyz is not None:
        return np.asarray(config.reference_xyz, dtype=float), 'kullanıcı referansı'
    return result.positions.iloc[-1][['x', 'y', 'z']].to_numpy(dtype=float), 'son epok çözümü'


def check_product_systems(products, systems, name):
    """Every selected system must be covered by the AR product (otherwise its ambiguities cannot be fixed)."""
    from gps_processing import gnss
    sats = products.frame.sat.to_numpy() if hasattr(products, 'frame') else np.array([], int)
    for system in systems:
        if not np.isin(gnss.systems_of(sats), [system]).any():
            raise ValueError(f'AR product {name} has no {gnss.NAMES[system]} biases. Choose a product that contains '
                             f'{gnss.NAMES[system]} or deselect {gnss.NAMES[system]}.')


def centre(path):
    """Analysis-centre prefix of an IGS long file name (GBM0MGXRAP_... -> GBM), else ''."""
    name = Path(str(path)).name
    return name[:3].upper() if len(name) > 20 and name[3:4].isdigit() and '_' in name else ''


def product_consistency(config):
    """(errors, warnings) about mixing analysis centres between the AR product and SP3/CLK."""
    errors, warnings = [], []
    if config.technique == 'PPP' or not config.ar_products:
        return errors, warnings
    product, orbit, clock = Path(config.ar_products), Path(str(config.sp3_day)).name, Path(str(config.clock_file)).name if config.clock_file else ''
    if is_osb(product):
        owner = centre(product)
        for kind, name in (('SP3', orbit), ('CLK', clock)):
            if owner and centre(name) and centre(name) != owner:
                errors.append(f'OSB product {product.name} is from {owner}, but the {kind} file {name} is from {centre(name)}. '
                              f'Use the {owner} SP3 and CLK files.')
    elif product.suffix.lower() == '.fcb':
        from gps_processing.io.ionex import read_text
        header = ' '.join(line for line in read_text(product)[:6] if line.startswith('#'))
        if 'orbit ' in header and 'clock ' in header:
            for kind, name in (('orbit', orbit), ('clock', clock or 'SP3')):
                if f'{kind} {name}' not in header:
                    errors.append(f'{product.name} was estimated with a different {kind} product than {name} (see its header).')
        else:
            warnings.append(f'{product.name} does not state which orbit/clock it was estimated with. Its narrow-lane FCBs are '
                            f'only valid with that product; make sure {orbit} and {clock or "the SP3 clocks"} are the ones its '
                            'provider used.')
    return errors, warnings


def is_osb(path):
    """Bias-SINEX (*.BIA, *.BSX, gz) AR products carry code and phase OSBs; others are JSON."""
    import re
    return bool(re.search(r'\.(BIA|BSX)(\.GZ)?$', str(path).upper()))


def settings(config):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}


def network_products(config, progress):
    """PPP-AR without a product file: estimate FCBs from the reference stations (Li & Zhang 2012)
    with the same orbit/clock/models and use the file of the best method (leave-one-station-out)."""
    if config.technique == 'PPP' or config.ar_products or len(config.fcb_files) < 5:
        return config, None
    from gps_processing.processing.fcb.core import METHODS
    from gps_processing.processing.fcb.network import estimate_fcb
    methods = METHODS if config.fcb_method == 'all' else (config.fcb_method,)
    step = lambda p, m: progress(int(0.5*p), f'[FCB] {m}')
    out = estimate_fcb(replace(config, technique='PPP', ar_products=''), config.fcb_files, methods, config.fcb_interval, progress=step)
    best = out['summary']['best_method'] or methods[0]
    path = out['output']/f'fcb_{best.replace("+", "_")}.fcb'
    progress(50, f'PPP-AR with the estimated FCBs ({best}): {path.name}')
    return replace(config, ar_products=str(path)), dict(out['summary'], product=str(path))


def run_directory(base, now=None):
    """New folder per process under ``base``: <year>_<day of year>_<HHMMSS> of the start time
    (e.g. results/2026_302_143512); a suffix keeps two runs in the same second apart."""
    from datetime import datetime
    now = now or datetime.now()
    target = Path(base)/f'{now:%Y}_{now:%j}_{now:%H%M%S}'
    k = 1
    while target.exists():
        k += 1
        target = Path(base)/f'{now:%Y}_{now:%j}_{now:%H%M%S}_{k}'
    target.mkdir(parents=True)
    return target


def run(config, progress=None):
    progress = progress or (lambda percent, message: None)
    config.validate()
    config = replace(config, output=run_directory(config.output))
    progress(1, f'Results of this process: {config.output}')
    errors, _ = product_consistency(config)
    if errors:
        raise ValueError('\n'.join(errors))
    config, network = network_products(config, progress)
    orbit, obs, satellites = prepare(config, progress)
    result = estimate(config, orbit, obs, satellites, progress)
    if network is not None:
        result.ar_metadata['fcb_estimation'] = network
    progress(90, 'Writing CSV, Excel and summary outputs')
    export_result(result, config.output)
    result.output_dir = config.output
    progress(100, f'Processing finished: {config.output}')
    return result


def compare(config, names, progress=None):
    """Run the named VARIANTS on the same prepared data; one output folder per variant."""
    progress = progress or (lambda percent, message: None)
    config.validate()
    config = replace(config, output=run_directory(config.output))
    names = [n for n in names if config.technique != 'PPP' or n not in AR_ONLY]
    if not names:
        raise ValueError('Karşılaştırma için en az bir uygun yöntem seçin (AR varyantları PPP-AR tekniği ister).')
    config, _ = network_products(config, progress)
    orbit, obs, satellites = prepare(config, progress)
    results, rows = {}, []
    for i, name in enumerate(names):
        overrides = dict(VARIANTS[name])
        if overrides.get('estimator') == 'sequential' and config.technique == 'PPP':
            overrides.update(ionosphere='', troposphere='', systems=('G',))   # the v2.0 baseline: GPS, no atmosphere models
        variant = replace(config, **overrides, output=Path(config.output)/'comparison'/slug(name))
        variant.validate()
        step = lambda p, m, i=i: progress(int(45+50*(i+p/100)/len(names)), f'[{name}] {m}')
        result = estimate(variant, orbit, obs, satellites, step)
        export_result(result, variant.output)
        results[name] = result
        rows.append({**result.statistics, 'solution': name})
    # Shared reference: user coordinate, otherwise the last epoch of the first variant.
    reference, source = reference_coordinate(config, results[names[0]])
    for row, name in zip(rows, names):
        results[name].positions = analysis.add_enu(results[name].positions, reference)
        row.update(analysis.statistics(results[name].positions, reference, name, source))
    table = pd.DataFrame(rows)
    target = Path(config.output)/'comparison'
    table.to_csv(target/'statistics.csv', index=False)
    table.attrs['output'] = Path(config.output)
    progress(100, f'Comparison finished: {config.output}')
    return table, results


def slug(name):
    name = name.translate(str.maketrans('çğıöşüÇĞİÖŞÜ', 'cgiosuCGIOSU'))
    return '_'.join(''.join(c if c.isalnum() else ' ' for c in name.encode('ascii', 'ignore').decode()).split()).lower()
