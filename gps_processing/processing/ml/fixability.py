"""Self-supervised ambiguity fixability from hindsight labels.

A first pass of the AR filter that fixes nothing gives, for every between-satellite arc, the
float WL and N1 ambiguities at every epoch. Where an arc ends converged (sigma < 0.05 cycle,
fraction < 0.10 cycle) its integer is known in hindsight; every earlier epoch of the arc is then
labelled "rounding now would give the right integer" or not. A gradient-boosted model learns this
probability from what is known in real time: sigma, fraction, Dong & Bock P0, elevation, arc
length and the stage (WL / N1). Models are cross-fitted by satellite (a satellite is never
scored by a model that saw its own labels).

The second pass uses the learned probability in the AR policy (processing.ar.policy):
* ordering and exclusion of candidates (generalising the P0 order of Ge et al. 2005), and
* a search trigger: the integer search starts only when at least ``ar_min_ambiguities``
  candidates are predicted fixable, the data-driven counterpart of the convergence indicator of
  PPP-ARISEN (CSI-ZWDV).
Arcs biased by inconsistent products or unmodelled effects never converge to integers, so the
model also learns not to fix them, which formal tests cannot see.
"""
import numpy as np
import pandas as pd
from gps_processing.processing.ar.integer import fix_probability
from gps_processing.processing.ml.learners import GradientBoostingRegressor
from gps_processing import gnss

FEATURES = ['log_sigma', 'abs_fraction', 'p0', 'sin_elevation', 'log_arc_epochs', 'stage_n1']


def features(value, sigma, elevation, arc_epochs, stage_n1):
    value, sigma = np.asarray(value, float), np.maximum(np.asarray(sigma, float), 1e-6)
    return np.column_stack([np.log(sigma), np.abs(value-np.rint(value)), fix_probability(value, sigma),
                            np.sin(np.radians(elevation)), np.log1p(arc_epochs), np.full(len(value), float(stage_n1))])


def arcs(ambiguities):
    """Arc number per row: a new arc starts when the arc age drops (slip) or the reference changes."""
    a = ambiguities.sort_values(['sat', 'epok']).copy()
    restart = (a.groupby('sat').arc_epochs.diff().fillna(-1) <= 0) | (a.groupby('sat').reference_sat.diff().fillna(1) != 0)
    a['arc'] = restart.groupby(a.sat).cumsum()
    return a


def hindsight_samples(ambiguities, final_sigma=0.05, final_fraction=0.10, every=5):
    """Features and labels (DataFrame) from a float AR pass (solver ambiguity table)."""
    a = ambiguities[ambiguities.sat != ambiguities.reference_sat]
    a = arcs(a[a.ar_usable.astype(bool)] if 'ar_usable' in a else a)
    rows = []
    for (sat, arc), g in a.groupby(['sat', 'arc']):
        last = g.iloc[-1]
        nw_truth = np.rint(last.NW_float)
        if last.sigma_NW > final_sigma or abs(last.NW_float-nw_truth) > final_fraction:
            continue
        g = g.iloc[::every]
        x = features(g.NW_float, g.sigma_NW, g.elevation, g.arc_epochs, 0)
        rows.append(pd.DataFrame(x, columns=FEATURES).assign(sat=sat, label=(np.rint(g.NW_float) == nw_truth).astype(float).to_numpy()))
        # N1 conditioned on the true wide lane (covariance from the filter).
        gain = g.cov_N1_NW/np.maximum(g.sigma_NW**2, 1e-12)
        n1 = g.N1_float+gain*(nw_truth-g.NW_float)
        sigma = np.sqrt(np.maximum(g.sigma_N1**2-g.cov_N1_NW**2/np.maximum(g.sigma_NW**2, 1e-12), 1e-12))
        last_gain = last.cov_N1_NW/max(last.sigma_NW**2, 1e-12)
        last_n1 = last.N1_float+last_gain*(nw_truth-last.NW_float)
        last_sigma = np.sqrt(max(last.sigma_N1**2-last.cov_N1_NW**2/max(last.sigma_NW**2, 1e-12), 1e-12))
        if last_sigma > final_sigma or abs(last_n1-np.rint(last_n1)) > final_fraction:
            # Converged but not integer: product or model error; every epoch is "not fixable".
            labels = np.zeros(len(g)) if last_sigma <= final_sigma else None
        else:
            labels = (np.rint(n1) == np.rint(last_n1)).astype(float).to_numpy()
        if labels is not None:
            x = features(n1, sigma, g.elevation, g.arc_epochs, 1)
            rows.append(pd.DataFrame(x, columns=FEATURES).assign(sat=sat, label=labels))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FEATURES+['sat', 'label'])


class FixabilityModel:
    """Cross-fitted (by satellite) gradient-boosted probability of a correct rounding."""

    def __init__(self, models, folds, info):
        self.models, self.folds, self.info = models, folds, info

    @classmethod
    def fit(cls, samples, folds=3, seed=0):
        if len(samples) < 200 or samples.label.nunique() < 2:
            return None
        sats = np.array(sorted(samples.sat.unique()))
        fold_of = {int(s): k % folds for k, s in enumerate(sats)}
        x, y = samples[FEATURES].to_numpy(float), samples.label.to_numpy(float)
        group = samples.sat.map(fold_of).to_numpy()
        models = []
        for k in range(folds):
            train = group != k
            models.append(GradientBoostingRegressor(n_estimators=80, max_depth=3, min_samples_leaf=20, seed=seed).fit(x[train], y[train]))
        info = dict(samples=int(len(y)), correct_fraction=float(y.mean()), satellites=int(len(sats)),
                    wl_samples=int((samples.stage_n1 == 0).sum()), n1_samples=int((samples.stage_n1 == 1).sum()),
                    n1_correct_fraction=float(y[samples.stage_n1.to_numpy() == 1].mean()) if (samples.stage_n1 == 1).any() else np.nan)
        return cls(models, fold_of, info)

    def probability(self, sats, x):
        """Probability per row; satellites unseen in training use the average of the fold models."""
        out = np.empty(len(x))
        for i, sat in enumerate(sats):
            k = self.folds.get(int(sat))
            models = [self.models[k]] if k is not None else self.models
            out[i] = np.mean([m.predict(x[i:i+1])[0] for m in models])
        return np.clip(out, 0.0, 1.0)

    def to_dict(self):
        return dict(info=self.info, features=FEATURES)


def learn(config, observations, satellites, approximate_xyz, variance_model=None, atmosphere=None):
    """First pass (float AR filter, no fixing) and the fitted model, or None with too few labels."""
    from dataclasses import replace
    from gps_processing.processing.ar.solver import solve
    float_pass = replace(config, ar_min_epochs=10**9, ar_fixability=False) if hasattr(config, '__dataclass_fields__') else config
    _, ambiguities, *_ = solve(observations, satellites, approximate_xyz, float_pass, variance_model, None, atmosphere)
    return FixabilityModel.fit(hindsight_samples(ambiguities))
