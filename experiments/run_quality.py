"""Experiment Q: outlier tests and robust estimation in float PPP (GPS + Galileo, kinematic, 2-h sessions).

Settings: no quality control, robust IGG-III / Huber / Danish, global chi-square test (flag only), Baarda data
snooping (B-method) and the Pope tau test, and Baarda followed by the Danish robust update.
Data: the real observations, and the same with simulated code blunders (1 % of the satellite-epochs, +-10..30 m)
whose identification is scored (detected = excluded by a test or inflated > 10 by a robust update).

    OMP_NUM_THREADS=1 python3 -m experiments.run_quality [stations ...]
Output: results/experiments/quality/quality.csv (one row per station, session, data, setting)
"""
import os
import sys
import multiprocessing as mp
from dataclasses import replace
import numpy as np
import pandas as pd
from experiments.common import OUT, enu_errors
from experiments.run_selection_rev2 import config
from gps_processing.pipeline import prepare, estimate

DAY = '2024081'
STATIONS = ['WTZR', 'GRAZ', 'BRUX', 'EBRE']
STARTS = (0.0, 28800.0, 57600.0)                 # 2-h sessions at 00, 08, 16 h
LENGTH = 7200.0
CONVERGED = 3600.0                               # s excluded from the accuracy statistics
TARGET = OUT/'quality'
SETTINGS = {
    'none': dict(robust=False, outlier_test='none'),
    'IGG-III': dict(robust=True, robust_method='igg3'),
    'Huber': dict(robust=True, robust_method='huber'),
    'Danish': dict(robust=True, robust_method='danish'),
    'global': dict(robust=False, outlier_test='global'),
    'Baarda': dict(robust=False, outlier_test='baarda'),
    'Pope': dict(robust=False, outlier_test='pope'),
    'Baarda+Danish': dict(robust=True, robust_method='danish', outlier_test='baarda'),
}


def blunders(combined, rate=0.01, seed=11):
    rng = np.random.default_rng(seed)
    pick = rng.random(len(combined)) < rate
    size = rng.uniform(10, 30, pick.sum())*rng.choice([-1, 1], pick.sum())
    out = combined.copy()
    out.loc[pick, 'Pc'] += size
    return out, set(zip(out.epok[pick].round(3), out.sat[pick].astype(int)))


def job(station):
    cfg, reference = config(station, DAY, ('G', 'E'))
    cfg = replace(cfg, dynamics='kinematic', position_noise=10.0)
    orbit, obs, satellites = prepare(cfg, lambda p, m: None)
    rows = []
    for start in STARTS:
        part = obs.combined[(obs.combined.epok >= start) & (obs.combined.epok < start+LENGTH)].reset_index(drop=True)
        bad, truth = blunders(part, seed=int(start)+len(station))
        for data, combined in (('real', part), ('blunders', bad)):
            for label, overrides in SETTINGS.items():
                try:
                    result = estimate(replace(cfg, **overrides), orbit, replace(obs, combined=combined), satellites)
                except Exception as exc:
                    rows.append(dict(station=station, start=start, data=data, setting=label, error=repr(exc)))
                    continue
                p = result.positions.sort_values('epok')
                enu = enu_errors(p, reference)
                late = (p.epok.to_numpy() >= start+CONVERGED)       # after the convergence period
                d3 = np.linalg.norm(enu, axis=1)
                row = dict(station=station, start=start, data=data, setting=label, epochs=len(p),
                           rms_3d_mm=float(np.sqrt(np.mean(d3[late]**2)))*1e3,
                           rms_u_mm=float(np.sqrt(np.mean(enu[late, 2]**2)))*1e3,
                           p95_3d_mm=float(np.percentile(d3[late], 95))*1e3, max_3d_mm=float(d3[late].max())*1e3)
                flagged = set()
                if 'qc_T' in p:
                    row.update(global_reject=float((~p.qc_first_passed.astype(bool)).mean()),
                               final_reject=float((~p.qc_passed.astype(bool)).mean()),
                               excluded_per_epoch=float(p.qc_excluded.mean()))
                    for t, sats in zip(p.epok, p.qc_excluded_sats.fillna('').astype(str)):
                        flagged |= {(round(float(t), 3), int(s)) for s in sats.split(',') if s}
                res = result.residuals
                if res is not None and 'inflation' in res:
                    flagged |= set(zip(res.epok[res.inflation > 10].round(3), res.sat[res.inflation > 10].astype(int)))
                used = set(zip(res.epok.round(3), res.sat.astype(int))) if res is not None else set()
                if data == 'blunders':
                    present = truth & used
                    row.update(blunders=len(present), detected=len(present & flagged)/max(len(present), 1),
                               false_flags=len(flagged-truth)/max(len(used), 1))
                else:
                    row.update(false_flags=len(flagged)/max(len(used), 1))
                rows.append(row)
        print(station, start, 'done', flush=True)
    return rows


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    stations = sys.argv[1:] or STATIONS
    with mp.get_context('fork').Pool(int(os.environ.get('NPROC', 2))) as pool:
        rows = [r for part in pool.map(job, stations, chunksize=1) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(TARGET/'quality.csv', index=False)
    cols = [c for c in ('rms_3d_mm', 'p95_3d_mm', 'max_3d_mm', 'global_reject', 'excluded_per_epoch', 'detected', 'false_flags')
            if c in df]
    print(df.groupby(['data', 'setting'], sort=False)[cols].mean().round(3).to_string())


if __name__ == '__main__':
    main()
