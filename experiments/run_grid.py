"""Run a grid of (configuration, session) solutions for MADR and collect metrics.

    python3 -m experiments.run_grid products      # Experiment A: PPP vs PPP-AR, four OSB product sets
    python3 -m experiments.run_grid ablation      # contribution of each v2.9 component (COD, SDBS)
    python3 -m experiments.run_grid network FCB   # Experiment B: FCBs estimated from the 50-station network

Each configuration is solved for the whole day and for twelve independent 2-h sessions
(restarted filter). Positions are written to results/experiments/<grid>/positions/ and the
metrics to results/experiments/<grid>/metrics.csv.
"""
import multiprocessing
import os
import sys
import time
from dataclasses import replace
import numpy as np
import pandas as pd
from experiments import common

STATE = {}


def prepare_centre(centre):
    from gps_processing.pipeline import prepare
    config = common.base_config(centre, 'PPP-AR/SDBS')
    orbit, obs, satellites = prepare(config, lambda p, m: None)
    return orbit, obs, satellites


def solve(job):
    name, centre, config, start = job
    from gps_processing.pipeline import estimate
    orbit, obs, satellites = STATE[centre]
    if start is not None:
        obs = replace(obs, combined=obs.combined[(obs.combined.epok >= start) &
                                                 (obs.combined.epok < start+common.SESSION)].reset_index(drop=True))
    t = time.time()
    try:
        result = estimate(config, orbit, obs, satellites)
    except Exception as exc:                                  # a failed session is recorded, not fatal
        return dict(name=name, centre=centre, technique=config.technique, session=start, error=str(exc)), None
    reference = np.asarray(config.reference_xyz, float)
    row = dict(name=name, centre=centre, technique=config.technique, session=-1 if start is None else int(start // 3600),
               seconds=time.time()-t, **common.metrics(result.positions, reference, 0.0 if start is None else start))
    keep = [c for c in ('epok', 'x', 'y', 'z', 'status', 'satellites', 'nis', 'gps_satellites', 'galileo_satellites',
                        'sigma_x', 'sigma_y', 'sigma_z', 'zwd_m') if c in result.positions]
    positions = result.positions[keep].assign(name=name, session=row['session'])
    return row, positions


def grid_products():
    jobs = []
    for centre in common.CENTRES:
        for technique in common.TECHNIQUES:
            config = common.base_config(centre, technique)
            jobs.append((f'{centre} {technique}', centre, config))
    return jobs


def grid_ablation(centre='COD', technique='PPP-AR/SDBS'):
    steps = [
        ('S0 v2.8 (ratio test, basic models)', dict(**common.V28_POLICY, **common.V28_MODELS)),
        ('S1 + physical models', dict(**common.V28_POLICY)),
        ('S2 + decision chain (P0, PAR, FFRT, exclusion)', {}),
        ('S2g + PRIDE candidate gates', dict(ar_candidate_gates=True)),
        ('S3 + adaptive class variance factors', dict(estimator='adaptive-ekf')),
        ('S4 + bias-aware ML acceptance', dict(estimator='adaptive-ekf', ar_validation='ffrt+ml')),
        ('S5 + self-supervised fixability', dict(estimator='adaptive-ekf', ar_fixability=True)),
        ('S6 + fix-and-hold', dict(estimator='adaptive-ekf', ar_fixability=True, ar_hold=True)),
    ]
    return [(name, centre, common.base_config(centre, technique, **kw)) for name, kw in steps] + \
           [('PPP float (v2.9 models)', centre, common.base_config(centre, 'PPP'))]


def grid_network(fcb_dir, centre='COD'):
    jobs = [('PPP float', centre, common.base_config(centre, 'PPP'))]
    for method in ('batch', 'ekf', 'ekf_ml'):
        path = os.path.join(fcb_dir, f'fcb_{method}.fcb')
        if not os.path.isfile(path):
            continue
        for technique in common.TECHNIQUES[1:]:
            jobs.append((f'FCB-{method} {technique}', centre, common.base_config(centre, technique, ar_products=path)))
    return jobs


def run(grid, jobs, workers=4):
    out = common.OUT/grid
    (out/'positions').mkdir(parents=True, exist_ok=True)
    for centre in sorted({j[1] for j in jobs}):
        print(f'preparing {centre}', flush=True)
        STATE[centre] = prepare_centre(centre)
    tasks = [(name, centre, config, start) for name, centre, config in jobs
             for start in [None]+[k*common.SESSION for k in range(int(86400//common.SESSION))]]
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    rows, done = [], 0
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        for row, positions in pool.imap_unordered(solve, tasks):
            done += 1
            rows.append(row)
            if positions is not None:
                tag = row['name'].replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                positions.to_csv(out/'positions'/f'{tag}_s{row["session"]}.csv', index=False)
            print(f'[{done}/{len(tasks)}] {row["name"]} session {row.get("session")} '
                  f'{"ERROR "+row["error"] if "error" in row else "fix %.2f conv %.1f" % (row["fix_rate"], row["convergence_min"])}', flush=True)
            pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    grid = sys.argv[1]
    jobs = {'products': grid_products, 'ablation': grid_ablation}.get(grid, lambda: grid_network(sys.argv[2]))()
    run(grid, jobs, workers=int(os.environ.get('WORKERS', 4)))
