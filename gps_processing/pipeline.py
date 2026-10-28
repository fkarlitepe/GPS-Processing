from dataclasses import replace, asdict
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


def prepare(config, progress):
    progress(5, 'Reading SP3 files')
    orbit = combine_sp3(config.sp3_before, config.sp3_day, config.sp3_after)
    progress(20, 'RINEX and signal combinations')
    obs = read_rinex(config.rinex)
    day = orbit.loc[orbit.epok == 0, 'timestamp']
    if day.empty or day.iloc[0].date() != obs.date.date():
        raise ValueError('RINEX gözlem günü ile ölçüm günü SP3 tarihi eşleşmiyor.')
    progress(40, 'Satellite clocks, 9th-order Lagrange interpolation and relativity')
    obs.combined, satellites = interpolate_satellites(orbit, obs.combined)
    return orbit, obs, satellites


def learn_weights(config, obs, satellites, progress, atmosphere=None):
    from gps_processing.processing import ppp_ekf
    from gps_processing.processing.ml.stochastic import ElevationModel, LearnedVarianceModel
    progress(55, 'ML learning pass: stochastic model from float EKF residuals')
    learning = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, config, atmosphere=atmosphere)
    return LearnedVarianceModel.fit(learning.residuals, ElevationModel(config.code_sigma, config.phase_sigma))


def estimate(config, orbit, obs, satellites, progress=None):
    progress = progress or (lambda percent, message: None)
    atmosphere = None
    if config.ionosphere or config.troposphere:
        from gps_processing.processing.atmosphere import Atmosphere
        progress(50, 'Reading external atmosphere products (IONEX / SINEX_TRO)')
        atmosphere = Atmosphere.from_config(config, obs)
    variance_model = learn_weights(config, obs, satellites, progress, atmosphere) if config.ml_weighting else None
    residuals, products, validator = None, None, None
    progress(65, f'{config.technique} · {config.estimator} estimation')
    if config.technique == 'PPP':
        if config.estimator == 'sequential':
            positions, ambiguities, diagnostics, covariance, corrections = solve(obs.combined, satellites, obs.approximate_xyz, config.elevation_mask)
        else:
            from gps_processing.processing import ppp_ekf
            output = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, config, variance_model, atmosphere)
            positions, ambiguities, diagnostics, covariance, corrections = output
            residuals = output.residuals
    else:
        from gps_processing.io.ar_products import ARProducts
        from gps_processing.processing.ar.solver import solve as solve_ar
        products = ARProducts(config.ar_products, config.technique.split('/')[-1], obs.date)
        if products.meta['orbit_product'] != Path(config.sp3_day).name:
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
    if atmosphere is not None:
        result.settings.update(atmosphere.describe())
    if products is not None:
        result.ar_metadata = {k: v for k, v in products.meta.items() if k != 'records'}
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


def settings(config):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}


def run(config, progress=None):
    progress = progress or (lambda percent, message: None)
    config.validate()
    orbit, obs, satellites = prepare(config, progress)
    result = estimate(config, orbit, obs, satellites, progress)
    progress(90, 'Writing CSV, Excel and summary outputs')
    export_result(result, config.output)
    progress(100, 'Processing finished')
    return result


def compare(config, names, progress=None):
    """Run the named VARIANTS on the same prepared data; one output folder per variant."""
    progress = progress or (lambda percent, message: None)
    config.validate()
    names = [n for n in names if config.technique != 'PPP' or n not in AR_ONLY]
    if not names:
        raise ValueError('Karşılaştırma için en az bir uygun yöntem seçin (AR varyantları PPP-AR tekniği ister).')
    orbit, obs, satellites = prepare(config, progress)
    results, rows = {}, []
    for i, name in enumerate(names):
        overrides = dict(VARIANTS[name])
        if overrides.get('estimator') == 'sequential' and config.technique == 'PPP':
            overrides.update(ionosphere='', troposphere='')   # the v2.0 baseline has no atmosphere models
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
    progress(100, 'Comparison finished')
    return table, results


def slug(name):
    name = name.translate(str.maketrans('çğıöşüÇĞİÖŞÜ', 'cgiosuCGIOSU'))
    return '_'.join(''.join(c if c.isalnum() else ' ' for c in name.encode('ascii', 'ignore').decode()).split()).lower()
