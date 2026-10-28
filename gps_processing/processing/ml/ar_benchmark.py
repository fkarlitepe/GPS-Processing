"""Benchmark of ambiguity acceptance tests on GNSS-like Monte Carlo data.

Every rule is compared at the same fixed failure rate (wrong fixes among accepted fixes,
the criterion of fixed-failure-rate ratio tests, Verhagen & Teunissen 2013): its threshold
is chosen on a calibration set and then applied to an independent test set. The score of a
rule is its availability (correctly accepted / all samples) on the test set.

    python3 -m gps_processing.processing.ml.ar_benchmark --output results/ar_benchmark
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .ar_simulation import simulate
from .ar_validation import CANDIDATES, build_model

FIXED_RULES = {
    'Ratio ≥ 3 (fixed)': lambda f: (np.exp(f[:, 0]) >= 3).astype(float),
    'Ps ≥ 0.999 + ratio ≥ 3 (fixed)': lambda f: ((sigmoid(f[:, 1]) >= 0.999) & (np.exp(f[:, 0]) >= 3)).astype(float),
}
CALIBRATED_SCORES = {'Ratio (FF-calibrated)': lambda f: f[:, 0], 'Success rate Ps (FF-calibrated)': lambda f: f[:, 1]}


def sigmoid(v):
    return 1/(1+np.exp(-v))


def threshold_for(score, label, failure_rate):
    """Smallest threshold whose accepted set has at most ``failure_rate`` wrong fixes."""
    order = np.argsort(-score)
    wrong = np.cumsum(1-label[order])/np.arange(1, len(order)+1)
    ok = np.flatnonzero(wrong <= failure_rate)
    return np.inf if len(ok) == 0 else float(score[order][ok[-1]])


def summary(name, accepted, label, extra=None):
    accepted = accepted.astype(bool)
    count = accepted.sum()
    row = dict(rule=name, availability=float((accepted & (label == 1)).mean()), acceptance=float(accepted.mean()),
               failure_rate=float((1-label[accepted]).mean()) if count else 0.0,
               missed_correct=float((~accepted & (label == 1)).sum()/max((label == 1).sum(), 1)))
    row.update(extra or {})
    return row


def evaluate_rules(models, x_cal, y_cal, x_test, y_test, failure_rate):
    """Rows for every rule at one target failure rate; ``oracle`` uses the test-set threshold
    (pure ranking quality), the other columns the calibration-set threshold (deployment)."""
    rows = [summary(name, rule(x_test), y_test) for name, rule in FIXED_RULES.items()]
    scores = {name: (score(x_cal), score(x_test)) for name, score in CALIBRATED_SCORES.items()}
    scores.update({name: (model.predict_proba(x_cal)[:, 1], model.predict_proba(x_test)[:, 1])
                   for name, model in models.items()})
    for name, (cal, test) in scores.items():
        cut = threshold_for(cal, y_cal, failure_rate)
        oracle = test >= threshold_for(test, y_test, failure_rate)
        rows.append(summary(name, test >= cut, y_test, dict(oracle_availability=float((oracle & (y_test == 1)).mean()))))
    for row in rows:
        row['target_failure_rate'] = failure_rate
    return rows


def run(train=12000, calibration=8000, test=8000, failure_rates=(0.003, 0.01), repeats=3, seed=0,
        candidates=None, progress=print):
    from sklearn.metrics import roc_auc_score
    rows, timing = [], []
    for repeat in range(repeats):
        base = seed+10*repeat
        progress(f'repeat {repeat+1}/{repeats}: simulating {train}+{calibration}+{test} samples')
        x_train, y_train = simulate(train, base)
        x_cal, y_cal = simulate(calibration, base+1)
        x_test, y_test = simulate(test, base+2)
        models = {}
        for name in candidates or CANDIDATES:
            start = time.perf_counter()
            models[name] = build_model(name).fit(x_train, y_train)
            fit_s = time.perf_counter()-start
            start = time.perf_counter()
            p = models[name].predict_proba(x_test)[:, 1]
            timing.append(dict(rule=name, fit_s=fit_s, predict_ms=(time.perf_counter()-start)/len(p)*1e3,
                               auc=roc_auc_score(y_test, p)))
        for rate in failure_rates:
            for row in evaluate_rules(models, x_cal, y_cal, x_test, y_test, rate):
                rows.append({**row, 'repeat': repeat})
        progress(f'repeat {repeat+1} done')
    detail = pd.DataFrame(rows)
    table = (detail.groupby(['target_failure_rate', 'rule'], sort=False)
             .agg(availability=('availability', 'mean'), availability_std=('availability', 'std'),
                  oracle_availability=('oracle_availability', 'mean'), failure_rate=('failure_rate', 'mean'),
                  acceptance=('acceptance', 'mean'))
             .reset_index())
    table = table.merge(pd.DataFrame(timing).groupby('rule').mean().reset_index(), on='rule', how='left')
    table = table.sort_values(['target_failure_rate', 'oracle_availability'], ascending=[True, False],
                              na_position='last').reset_index(drop=True)
    return table, detail


def best_model(table):
    """Candidate with the highest mean test availability at the deployment threshold, averaged over targets."""
    ml = table[table.rule.isin(CANDIDATES)]
    return ml.groupby('rule').availability.mean().idxmax()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=Path('results/ar_benchmark'))
    parser.add_argument('--train', type=int, default=12000)
    parser.add_argument('--calibration', type=int, default=8000)
    parser.add_argument('--test', type=int, default=8000)
    parser.add_argument('--failure-rates', type=float, nargs='+', default=[0.003, 0.01])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    table, detail = run(args.train, args.calibration, args.test, args.failure_rates, args.repeats)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output/'ar_validator_benchmark.csv', index=False)
    detail.to_csv(args.output/'ar_validator_benchmark_runs.csv', index=False)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'best model: {best_model(table)}')


if __name__ == '__main__':
    main()
