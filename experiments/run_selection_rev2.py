"""Experiment S, second revision: satellite selection in float PPP over three days (2023-11-23, 2024-03-21,
2024-06-21), with GPS + Galileo (GE) and GPS + Galileo + BDS (GEC).

Six independent 2-h sessions per station-day (starting 00, 04, ..., 20 h).
GE   k = 6, 8, 10: FIBO-GNSS2 with the four weightings and BACKWARD+LS with DOP and filter WDOP (threshold rule
     h = 0.05), RANDOM held 600 s, FIBO-GNSS2 held 600 s (DOP, filter WDOP), reused previous subset, and the filter left by the
     unselected satellites ('drop'); at k = 8 also the threshold h (0, 0.10), the holding time (300, 1800 s) and
     the first FIBO-GNSS+LS.
GEC  k = 8, 12, 16: FIBO-GNSS2 and BACKWARD+LS (DOP, filter WDOP), RANDOM held 600 s, FIBO-GNSS2 held 600 s and
     'drop'; kinematic PPP at k = 12.
timing  YEBE and WTZR, all days: one process on an otherwise idle computer, GE and GEC, keep and drop.

    OMP_NUM_THREADS=1 python3 -m experiments.run_selection_rev2 run [GE|GEC] [stations ...]
    OMP_NUM_THREADS=1 python3 -m experiments.run_selection_rev2 timing
Output: results/experiments/selection_rev2/{sessions_GE,sessions_GEC,timing}.csv
"""
import os
import sys
import time
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.common import DATA, OUT, metrics, sessions
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import prepare, estimate

DAYS = ('2023327', '2024081', '2024173')
STATIONS = {'GE': ['MADR', 'AJAC', 'BRUX', 'EBRE', 'ENAO', 'GRAZ', 'WTZR', 'SOFI'],
            'GEC': ['YEBE', 'AJAC', 'BRUX', 'EBRE', 'ENAO', 'GRAZ', 'WTZR', 'SOFI']}   # MADR tracks no BDS: YEBE
TARGET = OUT/'selection_rev2'
H = 0.05


def products(day):
    p = DATA/'products'/day/'COD'
    y, d = int(day[:4]), int(day[4:])
    name = lambda doy, kind: str(p/f'COD0MGXFIN_{y}{doy:03d}0000_01D_{kind}')
    sinex = (DATA/'sinex'/'IGS0OPSSNX_20233270000_01D_01D_CRD.SNX' if day == '2023327' else
             DATA/f'sinex_{day}'/f'IGS0OPSSNX_{day}0000_01D_01D_CRD.SNX')
    erp = DATA/'models'/'COD0MGXFIN_20233270000_01D_12H_ERP.ERP' if day == '2023327' else p/f'COD0MGXFIN_{day}0000_01D_12H_ERP.ERP'
    return dict(sp3_before=name(d-1, '05M_ORB.SP3'), sp3_day=name(d, '05M_ORB.SP3'), sp3_after=name(d+1, '05M_ORB.SP3'),
                clock_file=name(d, '30S_CLK.CLK'), attitude=name(d, '30S_ATT.OBX'), erp=str(erp)), sinex


def sinex_xyz(station, path):
    values = {}
    for line in open(path):
        if line[7:11] in ('STAX', 'STAY', 'STAZ') and line[14:18] == station:
            values[line[7:11]] = float(line[47:68])
    return np.array([values['STAX'], values['STAY'], values['STAZ']])


def rinex_of(station, day):
    if station == 'MADR':
        return DATA/'obs'/f'MADR00ESP_R_{day}0000_01D_30S_MO.rnx'
    folders = [DATA/'network'] if day == '2023327' else [DATA/f'network_{day}']
    return [f for folder in folders for f in sorted(folder.glob(f'{station}*_MO.rnx'))][0]


def config(station, day, systems):
    files, sinex = products(day)
    reference = sinex_xyz(station, sinex)
    return ProcessingConfig(rinex=str(rinex_of(station, day)), output=str(TARGET/'runs'), technique='PPP',
                            systems=systems, antex=str(DATA/'models'/'igs20_2287.atx'),
                            ocean_loading=str(DATA/'models'/'igs_stations_fes.blq'), reference_xyz=tuple(reference),
                            **files), reference


def sel(method, k, weighting, **kw):
    out = dict(selection=method, selection_count=k, selection_weighting=weighting, selection_hysteresis=H)
    out.update(kw)
    return out


def settings_ge(k):
    if k == 0:
        return [('ALL', dict(selection='all', selection_analysis=True))]
    runs = [(f'FIBO-GNSS2|{w}', sel('fibo-gnss2', k, w)) for w in ('none', 'elevation', 'cn0', 'filter')]
    runs += [(f'BACKWARD+LS|{w}', sel('backward+ls', k, w)) for w in ('none', 'filter')]
    hold = dict(selection_hysteresis=0.0, selection_interval=600.0)
    runs += [('RANDOM|none|hold600', sel('random', k, 'none', **hold)),
             ('FIBO-GNSS2|none|hold600', sel('fibo-gnss2', k, 'none', **hold)),
             ('FIBO-GNSS2|filter|hold600', sel('fibo-gnss2', k, 'filter', **hold)),
             ('FIBO-GNSS2|filter|reuse', sel('fibo-gnss2', k, 'filter', selection_warm_start=True)),
             ('FIBO-GNSS2|filter|drop', sel('fibo-gnss2', k, 'filter', selection_state='drop')),
             ('FIBO-GNSS2|filter|hold600|drop', sel('fibo-gnss2', k, 'filter', selection_state='drop', **hold))]
    if k == 8:
        runs += [(f'FIBO-GNSS2|filter|h={h}', sel('fibo-gnss2', k, 'filter', selection_hysteresis=h)) for h in (0.0, 0.10)]
        runs += [(f'FIBO-GNSS2|filter|hold{t}', sel('fibo-gnss2', k, 'filter', selection_hysteresis=0.0, selection_interval=float(t)))
                 for t in (300, 1800)]
        runs += [('FIBO-GNSS+LS|filter', sel('fibo-gnss+ls', k, 'filter'))]
    return runs


def settings_gec(k):
    if k == 0:
        return [('ALL', dict(selection='all', selection_analysis=True)),
                ('ALL|kinematic', dict(selection='all', selection_analysis=True, dynamics='kinematic'))]
    hold = dict(selection_hysteresis=0.0, selection_interval=600.0)
    runs = [(f'{m.upper()}|{w}', sel(m, k, w)) for m in ('fibo-gnss2', 'backward+ls') for w in ('none', 'filter')]
    runs += [('RANDOM|none|hold600', sel('random', k, 'none', **hold)),
             ('FIBO-GNSS2|filter|hold600', sel('fibo-gnss2', k, 'filter', **hold)),
             ('FIBO-GNSS2|filter|drop', sel('fibo-gnss2', k, 'filter', selection_state='drop'))]
    if k == 12:
        runs += [(f'{label}|kinematic', dict(kw, dynamics='kinematic')) for label, kw in
                 (('FIBO-GNSS2|none', sel('fibo-gnss2', k, 'none')), ('FIBO-GNSS2|filter', sel('fibo-gnss2', k, 'filter')),
                  ('FIBO-GNSS2|filter|hold600', sel('fibo-gnss2', k, 'filter', **hold)))]
    return runs


KS = {'GE': (0, 6, 8, 10), 'GEC': (0, 8, 12, 16)}
SETTINGS = {'GE': settings_ge, 'GEC': settings_gec}


def summary(result, reference, start):
    m = metrics(result.positions, reference, t0=start)
    e = result.selection['epochs']
    col = lambda c: float(e[c].mean()) if c in e else np.nan
    return dict(**m, rms_3d=float(np.sqrt(m['rms_e']**2+m['rms_n']**2+m['rms_u']**2)),
                mean_visible=col('visible'), mean_used=col('selected'), reselections=int(e.reselected.sum()) if 'reselected' in e else 0,
                selection_ms=col('selection_ms'), core_ms=col('core_ms'), update_ms=col('update_ms'),
                filter_states=col('filter_states'), evaluations=col('evaluations'),
                low_selected=col('low_selected'), low_all=col('low_all'),
                mean_el_selected=col('mean_elevation_selected'), mean_el_all=col('mean_elevation_all'),
                median_pdop=float(e.pdop_selected.median()), median_wpdop=float(e.wpdop_selected.median()),
                median_pdop_all=float(e.pdop_all.median()), median_wpdop_all=float(e.wpdop_all.median()),
                mean_sigma_u=float(result.positions.sigma_z.iloc[-240:].mean()))


def run_job(args):
    systems, day, station, k = args
    part = TARGET/'parts'/f'{systems}_{day}_{station}_k{k}.pkl'
    if part.exists():
        return pd.read_pickle(part)
    t0 = time.time()
    try:
        cfg, reference = config(station, day, tuple(systems))
        orbit, obs, satellites = prepare(cfg, lambda p, m: None)
    except Exception as exc:
        print(f'{systems} {day} {station} k={k}: FAILED {exc!r}', flush=True)
        return pd.DataFrame([dict(systems=systems, day=day, station=station, k=k, error=repr(exc))])
    rows = []
    for label, overrides in SETTINGS[systems](k):
        variant = replace(cfg, **overrides)
        variant.validate()
        for start, piece in sessions(obs):
            if piece.combined.epok.nunique() < 120 or int(start//7200) % 2:
                continue                                     # six independent 2-h sessions per day (00, 04, ... 20 h)
            base = dict(systems=systems, day=day, station=station, k=k, setting=label, session=int(start//7200))
            try:
                result = estimate(variant, orbit, piece, satellites)
                rows.append(dict(base, **summary(result, reference, start)))
            except Exception as exc:
                rows.append(dict(base, error=repr(exc)))
    out = pd.DataFrame(rows)
    out.to_pickle(part)
    print(f'{systems} {day} {station} k={k}: {len(rows)} runs in {time.time()-t0:.0f} s', flush=True)
    return out


def timing():
    """Run time on an idle computer: one process, YEBE and WTZR, all days and sessions."""
    rows = []
    for systems, ks in (('GE', (6, 8, 10)), ('GEC', (8, 12, 16))):
        for day in DAYS:
            for station in ('YEBE', 'WTZR'):
                cfg, reference = config(station, day, tuple(systems))
                orbit, obs, satellites = prepare(cfg, lambda p, m: None)
                runs = [('ALL', dict(selection='all', selection_analysis=True)),
                        ('ALL|full-state', dict(selection='all', selection_analysis=True, active_states=False))]
                for k in ks:
                    for method in ('fibo-gnss2', 'backward+ls'):
                        runs.append((f'{method.upper()}|filter', dict(sel(method, k, 'filter'), _k=k)))
                    runs.append(('FIBO-GNSS2|filter|drop', dict(sel('fibo-gnss2', k, 'filter', selection_state='drop'), _k=k)))
                    runs.append(('FIBO-GNSS2|filter|hold600|drop', dict(sel('fibo-gnss2', k, 'filter', selection_state='drop',
                                                                            selection_hysteresis=0.0, selection_interval=600.0), _k=k)))
                for label, overrides in runs:
                    k = overrides.pop('_k', 0)
                    variant = replace(cfg, **overrides)
                    for start, piece in sessions(obs):
                        if piece.combined.epok.nunique() < 120 or int(start//7200) % 2:
                            continue
                        result = estimate(variant, orbit, piece, satellites)
                        e = result.selection['epochs']
                        rows.append(dict(systems=systems, day=day, station=station, k=k, setting=label,
                                         session=int(start//7200), visible=e.visible.mean(), used=e.selected.mean(),
                                         states=e.filter_states.mean(), selection_ms=e.selection_ms.mean() if k else 0.0,
                                         core_ms=e.core_ms.mean(), update_ms=e.update_ms.mean(),
                                         evaluations=e.evaluations.mean()))
                print(systems, day, station, 'timing done', flush=True)
    pd.DataFrame(rows).to_csv(TARGET/'timing.csv', index=False)


def validate_job(args):
    systems, day, station = args
    try:
        cfg, reference = config(station, day, tuple(systems))
        orbit, obs, satellites = prepare(cfg, lambda p, m: None)
        result = estimate(replace(cfg, selection='all'), orbit, obs, satellites)
    except Exception as exc:
        return dict(systems=systems, day=day, station=station, error=repr(exc))
    from experiments.common import enu_errors
    p = result.positions.sort_values('epok')
    enu = enu_errors(p, reference)
    last = p.epok.to_numpy() >= p.epok.max()-6*3600
    return dict(systems=systems, day=day, station=station, final_e=enu[-1, 0]*1e3, final_n=enu[-1, 1]*1e3,
                final_u=enu[-1, 2]*1e3, final_3d=float(np.linalg.norm(enu[-1]))*1e3,
                rms_last6h_3d=float(np.sqrt(np.mean(np.sum(enu[last]**2, 1))))*1e3)


def validate():
    """24-h static all-in-view PPP of every station-day against the IGS daily SINEX (software check)."""
    jobs = [(sy, day, st) for sy in ('GE', 'GEC') for day in DAYS for st in STATIONS[sy]]
    with mp.get_context('fork').Pool(int(os.environ.get('NPROC', 4))) as pool:
        rows = pool.map(validate_job, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(TARGET/'validation_24h.csv', index=False)
    print(pd.DataFrame(rows).groupby('systems')[['final_3d', 'rms_last6h_3d']].describe().round(1))


def main():
    (TARGET/'parts').mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == 'timing':
        return timing()
    if sys.argv[1] == 'validate':
        return validate()
    systems = sys.argv[2]
    stations = sys.argv[3:] or STATIONS[systems]
    jobs = [(systems, day, s, k) for day in DAYS for s in stations for k in KS[systems]]
    jobs.sort(key=lambda j: -len(SETTINGS[j[0]](j[3])))          # long jobs first
    with mp.get_context('fork').Pool(int(os.environ.get('NPROC', 6))) as pool:
        out = pool.map(run_job, jobs, chunksize=1)
    rows = pd.concat(out, ignore_index=True)
    rows.to_csv(TARGET/f'sessions_{systems}.csv', index=False)
    print('saved', len(rows), 'session runs ->', TARGET)


if __name__ == '__main__':
    main()
