"""Narrow-lane diagnostic: are the float between-satellite N1 ambiguities close to integers?

Following the literature check (Dai 2019: NL residual STD ~0.07 cycle; Ge 2008: 91 % within
+-0.1 cycle), the station is processed with its coordinates held at a known value (a SINEX
coordinate, or the converged float PPP of the same day), so that position errors do not leak
into the ambiguities, and no integer is fixed. For every converged arc (sigma < 0.05 cycle after
conditioning on the integer wide lane, arc > 30 min) the N1 fraction is sampled every 15 min.

* fractions clustered near 0                 -> products/models are consistent; NL failures in
                                                normal runs come from convergence or validation;
* a broad (uniform-like) distribution        -> product inconsistency (biases of one analysis
                                                centre with the clocks of another, wrong ANTEX,
                                                code signals) or unmodelled cm-level effects;
* a few satellites with large mean fractions -> satellite-specific errors (attitude/eclipse,
                                                missing PCV, bad bias).
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gps_processing import gnss


def wrap(value):
    return (np.asarray(value, float)+0.5) % 1.0-0.5


def narrow_lane_fractions(ambiguities, interval=30.0, min_arc_s=1800.0, max_sigma=0.05, wl_gate=0.15, sample_s=900.0):
    """Samples (DataFrame) of conditioned SD N1 fractions from the AR solver ambiguity table."""
    a = ambiguities[(ambiguities.sat != ambiguities.reference_sat) & np.isfinite(ambiguities.N1_float)].copy()
    if 'ar_usable' in a:
        a = a[a.ar_usable.astype(bool)]                  # satellites the product covers
    a = a[(a.arc_epochs*interval >= min_arc_s) & (np.isclose(a.epok % sample_s, 0))]
    if a.empty:
        return a.assign(wl_fraction=[], n1=[], sigma=[], fraction=[])
    k = np.array([gnss.constants(gnss.system_of(s))['beta']/gnss.constants(gnss.system_of(s))['lambda_n'] for s in a.sat])
    wl = np.rint(a.NW_float.to_numpy())
    a['wl_fraction'] = a.NW_float.to_numpy()-wl
    # Conditioning on the integer wide lane: N1 | NW = N1 + (cov/var_NW)(NW_int - NW); for the IF
    # ambiguity alone this is N1 + (beta/lambda_n)(NW - NW_int).
    variance_nw = np.maximum(a.sigma_NW.to_numpy()**2, 1e-12)
    gain = a.cov_N1_NW.to_numpy()/variance_nw
    a['n1'] = a.N1_float.to_numpy()+gain*(wl-a.NW_float.to_numpy())
    a['sigma'] = np.sqrt(np.maximum(a.sigma_N1.to_numpy()**2-a.cov_N1_NW.to_numpy()**2/variance_nw, 0))
    a['if_fraction'] = wrap(a.N1_float.to_numpy()+k*(a.NW_float.to_numpy()-wl))
    a = a[(np.abs(a.wl_fraction) < wl_gate) & (a.sigma < max_sigma)]
    a['fraction'] = wrap(a.n1.to_numpy())
    return a


def summarize(samples):
    if samples.empty:
        return dict(samples=0)
    f = np.abs(samples.fraction.to_numpy())
    per_sat = samples.groupby('sat').fraction.agg(lambda v: float(np.angle(np.mean(np.exp(2j*np.pi*v)))/(2*np.pi)))
    worst = per_sat.abs().sort_values(ascending=False)
    return dict(samples=int(len(f)), arcs=int(samples.groupby(['sat', 'reference_sat']).ngroups),
                median_abs_fraction=float(np.median(f)), rms_fraction=float(np.sqrt(np.mean(f**2))),
                within_010=float(np.mean(f < 0.10)), within_015=float(np.mean(f < 0.15)), within_025=float(np.mean(f < 0.25)),
                uniform_reference=dict(median_abs_fraction=0.25, within_010=0.20, within_015=0.30),
                literature_reference='Dai 2019: STD ~0.07 cycle; Ge 2008: 91 % within 0.1 cycle',
                worst_satellites={gnss.label(int(s)): round(float(per_sat[s]), 3) for s in worst.index[:6]},
                histogram=np.histogram(samples.fraction, bins=10, range=(-0.5, 0.5))[0].tolist())


def verdict(summary):
    if not summary.get('samples'):
        return 'no converged arcs: process a longer file (>= 6 h) or check the products'
    if summary['within_015'] >= 0.7:
        return 'consistent: fractions cluster at integers; remaining NL failures are convergence/validation issues'
    if summary['within_015'] <= 0.4:
        return ('inconsistent: fractions are nearly uniform; check that the phase biases, clocks and orbits come from the same '
                'analysis centre (file headers), the ANTEX version and the code signals')
    return 'partially consistent: look at the worst satellites (attitude, PCV, bias) and the unmodelled effects'


def converged_coordinate(config, orbit, obs, satellites, progress):
    """Float PPP of the same day; mean of the last hour as the held coordinate."""
    from gps_processing.pipeline import estimate
    float_config = replace(config, technique='PPP', estimator='ekf' if config.estimator == 'sequential' else config.estimator,
                           ml_weighting=False, reference_xyz=None, hold_coordinates=False)
    result = estimate(float_config, orbit, obs, satellites, progress)
    last = result.positions[result.positions.epok >= result.positions.epok.max()-3600]
    return tuple(last[['x', 'y', 'z']].mean().to_numpy(float))


def run_nl_diagnostic(config, variants=None, progress=None):
    """Hold the coordinates, run the AR filter without fixing for each variant and compare the
    N1 fraction distributions. ``variants``: {name: config overrides}; default compares the
    current settings with the extra models switched off. Writes <output>/nl_diagnostic/."""
    from gps_processing.pipeline import prepare, estimate, run_directory
    progress = progress or (lambda p, m: None)
    if config.technique == 'PPP':
        raise ValueError('The narrow-lane diagnostic needs a PPP-AR technique and an AR product.')
    config.validate()
    config = replace(config, output=run_directory(config.output))
    target = Path(config.output)/'nl_diagnostic'
    target.mkdir(parents=True, exist_ok=True)
    orbit, obs, satellites = prepare(config, progress)
    if config.reference_xyz is None:
        progress(45, 'Converged float coordinate (no reference given)')
        reference, source = converged_coordinate(config, orbit, obs, satellites, progress), 'converged float PPP (last hour)'
    else:
        reference, source = tuple(config.reference_xyz), 'user reference'
    variants = variants or {
        'selected models': {},
        'without ocean/pole/Shapiro/attitude/sat-PCV': dict(ocean_loading='', erp='', shapiro=False, attitude='', satellite_pcv=False),
    }
    interval = float(np.median(np.diff(np.unique(obs.combined.epok)))) if obs.combined.epok.nunique() > 1 else 30.0
    report = dict(reference_xyz=list(reference), reference_source=source, ar_products=str(config.ar_products), variants={})
    for k, (name, overrides) in enumerate(variants.items()):
        progress(50+int(45*k/len(variants)), f'NL diagnostic: {name}')
        variant = replace(config, **overrides, reference_xyz=reference, hold_coordinates=True, ar_min_epochs=10**9)
        result = estimate(variant, orbit, obs, satellites)
        samples = narrow_lane_fractions(result.ambiguities, interval)
        slug = ''.join(c if c.isalnum() else '_' for c in name).strip('_')
        samples.to_csv(target/f'n1_fractions_{slug}.csv', index=False)
        summary = summarize(samples)
        summary['verdict'] = verdict(summary)
        summary['models'] = {key: result.settings.get(key) for key in ('ocean_loading', 'pole_tide', 'shapiro_delay', 'attitude',
                                                                       'satellite_pcv', 'solid_earth_tide') if key in result.settings}
        report['variants'][name] = summary
    (target/'summary.json').write_text(json.dumps(report, indent=2, default=float))
    progress(100, f'NL diagnostic finished: {target}')
    report['output'] = str(target)
    return report
