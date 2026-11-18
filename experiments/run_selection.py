"""Experiment S: satellite selection in float PPP, DOP vs weighted DOP (WDOP).

For every station, 2-h session (12 per day) and setting, float PPP (GPS + Galileo, COD MGEX final
orbit/clock, v2.9 models, static EKF) is run with only k satellites per update:

* ALL       all visible satellites (reference; DOP analysis only)
* methods   FIBO-GNSS+LS (Fibonacci sphere candidates + 1-swap local search), BACKWARD+LS (greedy backward
            elimination + local search), TOP-CN0 and RANDOM (lower bound)
* weights   none (classical PDOP), elevation (sin^2 el), cn0 (C/N0) and filter (the EKF's own code
            variances: the geometry the filter actually weights)  -> the WDOP effect
* k         6, 8, 10;   hysteresis 5 % (a new subset must lower the objective by 5 %)
* ablation  FIBO-GNSS+LS, filter weights, k = 8 without hysteresis (re-selection every epoch)

Output: results/experiments/selection/sessions.csv (one row per station, session and setting) and
selection_epochs.csv.gz (per-epoch DOP of every run), analysed by experiments/analyse_selection.py.

    OMP_NUM_THREADS=1 python3 -m experiments.run_selection [stations ...]
"""
import os
import sys
import time
import multiprocessing as mp
from dataclasses import replace
import numpy as np
import pandas as pd
from experiments.common import DATA, OUT, RINEX, base_config, sinex_xyz, sessions, metrics
from gps_processing.pipeline import prepare, estimate

STATIONS = ['MADR', 'AJAC', 'BRUX', 'EBRE', 'ENAO', 'GRAZ', 'WTZR', 'SOFI']
KS = (6, 8, 10)
METHODS = ('fibo-gnss+ls', 'backward+ls')
WEIGHTINGS = ('none', 'elevation', 'cn0', 'filter')
HYSTERESIS = 0.05
TARGET = OUT/'selection'


def rinex_of(station):
    if station == 'MADR':
        return RINEX
    files = sorted((DATA/'network').glob(f'{station}*_MO.rnx'))
    if not files:
        raise FileNotFoundError(f'{station}: no RINEX in Data/network')
    return files[0]


def settings(k):
    """(label, overrides) of the runs at subset size k (k = 0: the all-in-view reference)."""
    if k == 0:
        return [('ALL', dict(selection='all', selection_analysis=True))]
    runs = [(f'{m.upper()}|{w}', dict(selection=m, selection_count=k, selection_weighting=w, selection_hysteresis=HYSTERESIS))
            for m in METHODS for w in WEIGHTINGS]
    runs += [('TOP-CN0|cn0', dict(selection='top-cn0', selection_count=k, selection_weighting='cn0', selection_hysteresis=HYSTERESIS)),
             ('RANDOM|none', dict(selection='random', selection_count=k, selection_weighting='none', selection_interval=600.0))]
    if k == 8:
        runs.append(('FIBO-GNSS+LS|filter|no-hysteresis', dict(selection='fibo-gnss+ls', selection_count=k, selection_weighting='filter')))
    return runs


def job(args):
    """One (station, k) job; written to parts/ so a crash or restart loses nothing."""
    station, k = args
    part_file = TARGET/'parts'/f'{station}_k{k}.pkl'
    if part_file.exists():
        return pd.read_pickle(part_file)
    try:
        out = run_job(station, k)
    except Exception as exc:                       # keep the other jobs alive
        print(f'{station} k={k}: FAILED {exc!r}', flush=True)
        return pd.DataFrame([dict(station=station, k=k, error=repr(exc))]), pd.DataFrame()
    pd.to_pickle(out, part_file)
    return out


def run_job(station, k):
    t0 = time.time()
    reference = sinex_xyz(station)
    config = base_config('COD', rinex=str(rinex_of(station)), reference_xyz=tuple(reference), output=str(TARGET/'runs'))
    orbit, obs, satellites = prepare(config, lambda p, m: None)
    rows, epochs = [], []
    for label, overrides in settings(k):
        variant = replace(config, **overrides)
        variant.validate()
        for start, part in sessions(obs):
            if part.combined.epok.nunique() < 120:
                continue
            try:
                result = estimate(variant, orbit, part, satellites)
            except ValueError as exc:
                rows.append(dict(station=station, k=k, setting=label, session=int(start//7200), error=str(exc)))
                continue
            m = metrics(result.positions, reference, t0=start)
            s = result.selection['summary']
            e = result.selection['epochs']
            method, weighting = (label.split('|') + [''])[:2]
            rows.append(dict(station=station, k=k, setting=label, method=method, weighting=weighting,
                             hysteresis='no-hysteresis' not in label and label != 'ALL', session=int(start//7200), **m,
                             mean_visible=s['mean_visible'], mean_used=s['mean_selected'], reselections=s['reselections'],
                             selection_ms=s['mean_selection_ms'], median_pdop=s['median_pdop_selected'],
                             median_wpdop=s['median_wpdop_selected'], median_hdop=s['median_hdop_selected'],
                             median_vdop=s['median_vdop_selected'], median_pdop_all=s['median_pdop_all'],
                             median_wpdop_all=s['median_wpdop_all'],
                             mean_sigma_u=float(result.positions.sigma_z.iloc[-240:].mean())))
            keep = ['epok', 'visible', 'selected', 'pdop_all', 'pdop_selected', 'wpdop_all', 'wpdop_selected',
                    'hdop_selected', 'vdop_selected', 'reselected']
            epochs.append(e[keep].assign(station=station, k=k, setting=label, session=int(start//7200)))
    print(f'{station} k={k}: {len(rows)} session runs in {time.time()-t0:.0f} s', flush=True)
    return pd.DataFrame(rows), pd.concat(epochs, ignore_index=True) if epochs else pd.DataFrame()


def main():
    stations = sys.argv[1:] or STATIONS
    (TARGET/'parts').mkdir(parents=True, exist_ok=True)
    jobs = [(s, k) for s in stations for k in (0,)+KS]
    with mp.get_context('fork').Pool(int(os.environ.get('NPROC', 7))) as pool:
        out = pool.map(job, jobs, chunksize=1)
    rows = pd.concat([o[0] for o in out], ignore_index=True)
    epochs = pd.concat([o[1] for o in out if len(o[1])], ignore_index=True)
    rows.to_csv(TARGET/'sessions.csv', index=False)
    epochs.to_csv(TARGET/'selection_epochs.csv.gz', index=False)
    print('saved', len(rows), 'session runs ->', TARGET)


if __name__ == '__main__':
    main()
