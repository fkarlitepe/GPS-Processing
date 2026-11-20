"""Experiment S, revision runs: subset stability, warm start, run time and selected-satellite elevations.

Same stations, sessions and models as experiments.run_selection; new settings (FIBO-GNSS+LS):

* hold600   the subset is kept for 600 s (no hysteresis), as the RANDOM (600 s) control, with classical DOP
            ('none') and with filter WDOP -> separates subset stability from the weighting
* warm      filter WDOP, 5 % hysteresis, previous subset as an extra FIBO-GNSS candidate (warm start)
* rerun     the main settings (hysteresis 5 %) re-run to log the elevation of the selected satellites:
            k = 6 and 8 with 'none' and 'filter', k = 8 also with 'elevation' and 'cn0'

With ``timing`` as the first argument, the ALL / FIBO-GNSS+LS / BACKWARD+LS (filter) runs of two stations are
processed one at a time (no parallel jobs) to measure the run time of selection and filter per epoch.

    OMP_NUM_THREADS=1 python3 -m experiments.run_selection_revision [timing] [stations ...]
"""
import os
import sys
import time
import multiprocessing as mp
from dataclasses import replace
import numpy as np
import pandas as pd
from experiments.common import OUT, base_config, sinex_xyz, sessions, metrics
from experiments.run_selection import STATIONS, rinex_of
from gps_processing.pipeline import prepare, estimate

TARGET = OUT/'selection_revision'
H = 0.05


def settings(k):
    fibo = dict(selection='fibo-gnss+ls', selection_count=k)
    runs = [(f'FIBO-GNSS+LS|{w}|hold600', dict(fibo, selection_weighting=w, selection_interval=600.0)) for w in ('none', 'filter')]
    runs.append(('FIBO-GNSS+LS|filter|warm', dict(fibo, selection_weighting='filter', selection_hysteresis=H, selection_warm_start=True)))
    rerun = ('none', 'filter') if k == 6 else ('none', 'elevation', 'cn0', 'filter') if k == 8 else ()
    runs += [(f'FIBO-GNSS+LS|{w}', dict(fibo, selection_weighting=w, selection_hysteresis=H)) for w in rerun]
    return runs


def timing_settings(k):
    if k == 0:
        return [('ALL', dict(selection='all', selection_analysis=True))]
    return [(f'{m.upper()}|filter', dict(selection=m, selection_count=k, selection_weighting='filter', selection_hysteresis=H))
            for m in ('fibo-gnss+ls', 'backward+ls')]


def run_job(args):
    station, k, timing = args
    target = TARGET/('timing' if timing else 'parts')
    part_file = target/f'{station}_k{k}.pkl'
    if part_file.exists():
        return pd.read_pickle(part_file)
    try:
        out = run(station, k, timing)
    except Exception as exc:
        print(f'{station} k={k}: FAILED {exc!r}', flush=True)
        return pd.DataFrame([dict(station=station, k=k, error=repr(exc))])
    out.to_pickle(part_file)
    return out


def run(station, k, timing):
    t0 = time.time()
    reference = sinex_xyz(station)
    config = base_config('COD', rinex=str(rinex_of(station)), reference_xyz=tuple(reference), output=str(TARGET/'runs'))
    orbit, obs, satellites = prepare(config, lambda p, m: None)
    rows = []
    for label, overrides in (timing_settings if timing else settings)(k):
        variant = replace(config, **overrides)
        variant.validate()
        for start, part in sessions(obs):
            if part.combined.epok.nunique() < 120:
                continue
            tick = time.perf_counter()
            try:
                result = estimate(variant, orbit, part, satellites)
            except ValueError as exc:
                rows.append(dict(station=station, k=k, setting=label, session=int(start//7200), error=str(exc)))
                continue
            wall = time.perf_counter()-tick
            m = metrics(result.positions, reference, t0=start)
            s = result.selection['summary']
            e = result.selection['epochs']
            row = dict(station=station, k=k, setting=label, session=int(start//7200), **m,
                       mean_visible=s['mean_visible'], mean_used=s['mean_selected'], reselections=s['reselections'],
                       median_pdop=s['median_pdop_selected'], median_wpdop=s['median_wpdop_selected'],
                       selection_ms=float(e.selection_ms.mean()), evaluations=float(e.evaluations.mean()),
                       core_ms=float(e.core_ms.mean()) if 'core_ms' in e else np.nan,
                       update_ms=float(e.update_ms.mean()) if 'update_ms' in e else np.nan,
                       wall_s=wall, n_epochs=len(e),
                       mean_el_selected=float(e.mean_elevation_selected.mean()),
                       mean_el_all=float(e.mean_elevation_all.mean()))
            sat = result.selection.get('satellites')
            if sat is not None and len(sat):
                used = sat[sat.selected.astype(bool)]
                row.update(share_low_selected=float((used.elevation < 30).mean()),
                           share_low_visible=float((sat.elevation < 30).mean()),
                           median_el_selected=float(used.elevation.median()))
            rows.append(row)
    print(f'{station} k={k}{" timing" if timing else ""}: {len(rows)} session runs in {time.time()-t0:.0f} s', flush=True)
    return pd.DataFrame(rows)


def main():
    args = sys.argv[1:]
    timing = bool(args) and args[0] == 'timing'
    if timing:
        stations = args[1:] or ['MADR', 'WTZR']
        (TARGET/'timing').mkdir(parents=True, exist_ok=True)
        out = [run_job((s, k, True)) for s in stations for k in (0, 6, 8, 10)]   # one at a time: clean timing
        pd.concat(out, ignore_index=True).to_csv(TARGET/'timing.csv', index=False)
        print('saved timing ->', TARGET/'timing.csv')
        return
    stations = args or STATIONS
    (TARGET/'parts').mkdir(parents=True, exist_ok=True)
    jobs = [(s, k, False) for k in (8, 6, 10) for s in stations]
    with mp.get_context('fork').Pool(int(os.environ.get('NPROC', 4))) as pool:
        out = pool.map(run_job, jobs, chunksize=1)
    rows = pd.concat(out, ignore_index=True)
    rows.to_csv(TARGET/'sessions.csv', index=False)
    print('saved', len(rows), 'session runs ->', TARGET)


if __name__ == '__main__':
    main()
