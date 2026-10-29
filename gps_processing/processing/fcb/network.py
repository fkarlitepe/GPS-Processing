"""FCB estimation from a reference network (at least five stations), Li & Zhang (2012) workflow.

1. Every station is processed with the float EKF PPP (same orbit, clock, ANTEX and models the
   users will apply); float zero-difference ambiguities are kept with their sigma and arc. With
   ``fcb_hold_coordinates`` (default) a first float pass gives the daily coordinate and the second
   pass holds it, as network solutions do (Geng 2012, PRIDE), so that position errors do not leak
   into the ambiguities. Stations run in parallel processes (100+ stations).
2. Wide lane: arc means of the MW combination (cycles) -> WL UFOs (one block per day by
   default) -> integer WL ambiguity of every arc whose fractional residual passes the gate.
3. Narrow lane: B_NL = (B_IF - beta*N_WL)/lambda_NL from the float IF ambiguity at the end of
   each block (default 15 min) -> NL UFOs per block.
4. The UFOs are written as GENFCB L1/L2 FCBs, b1 = s_NL - (beta/lambda_NL)*s_WL and
   b2 = b1 - s_WL, which io.fcb_products turns back into the same WL and IF corrections.
Each estimator (batch / EKF / EKF+ML) is validated by leaving each station out in turn (an
evenly spread subset of 20 stations in large networks). Satellite FCBs of a block observed by fewer
than ``fcb_min_contributors`` stations are dropped (0 = stations/10 within [1, 10]; Geng et al. 2012
use 10 in global networks; with 5 sparse global stations a filter of 3 left 37 FCBs for the day).
Literature networks: 80-200 global stations, or >= 10 regional stations within ~2000 km of the user
(Li & Zhang 2012); the summary warns when the network is smaller or farther away.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
from pathlib import Path
import json
import multiprocessing
import os
import numpy as np
import pandas as pd
from gps_processing.processing.ar.models import BETA, LAMBDA_N, LAMBDA_W
from gps_processing.processing.fcb import core
from gps_processing import gnss

MIN_STATIONS = 5


def validate_stations(stations):
    from gps_processing.data_directory import require_data_file
    paths = [require_data_file(p) for p in stations]
    if len(set(paths)) < MIN_STATIONS:
        raise ValueError(f'FCB kestirimi için en az {MIN_STATIONS} farklı istasyon RINEX dosyası gerekli ({len(set(paths))} verildi).')
    return paths


def station_ambiguities(config, path, progress=lambda p, m: None):
    """Float PPP of one station -> (name, date, float IF ambiguities per epoch with MW, coordinate)."""
    from gps_processing.pipeline import prepare, antenna_models
    from gps_processing.processing import ppp_ekf
    from gps_processing.processing.atmosphere import Atmosphere
    station = replace(config, rinex=path, technique='PPP', estimator='ekf' if config.estimator == 'sequential' else config.estimator,
                      ml_weighting=False, troposphere='', reference_xyz=None, hold_coordinates=False, dynamics='static')
    orbit, obs, satellites = prepare(station, progress)
    obs, satellites, _ = antenna_models(station, obs, satellites)
    atmosphere = Atmosphere.from_config(station, obs)
    output = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, station, atmosphere=atmosphere)
    last = output.positions[output.positions.epok >= output.positions.epok.max()-3600]
    coordinate = last[['x', 'y', 'z']].mean().to_numpy(float)
    if getattr(config, 'fcb_hold_coordinates', True):
        held = replace(station, reference_xyz=tuple(coordinate), hold_coordinates=True)
        output = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, held, atmosphere=atmosphere)
    name = (obs.marker or Path(path).name)[:9].strip() or Path(path).stem[:4]
    ambiguities = output.ambiguities.merge(obs.combined[['epok', 'sat', 'A4']], on=['epok', 'sat'], how='left')
    return name, obs.date, ambiguities, coordinate


def process_stations(config, paths, progress, workers=0):
    """station_ambiguities for every path, in parallel processes (``workers`` 0: CPU count, max 8).
    BLAS threads are limited to one per process to avoid oversubscription."""
    workers = workers or min(os.cpu_count() or 1, 8, len(paths))
    if workers <= 1:
        out = []
        for k, path in enumerate(paths):
            progress(5+int(55*k/len(paths)), f'Float PPP for FCB: station {k+1}/{len(paths)} ({path.name})')
            out.append(station_ambiguities(config, path))
        return out
    saved = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    os.environ.update({key: '1' for key in saved})
    try:
        results = [None]*len(paths)
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(station_ambiguities, config, path): k for k, path in enumerate(paths)}
            for done, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                progress(5+int(55*done/len(paths)), f'Float PPP for FCB: {done}/{len(paths)} stations ({workers} processes)')
        return results
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def network_warnings(coordinates, user_xyz=None):
    """Literature checks of the network geometry (Li & Zhang 2012; Geng et al. 2012)."""
    warnings = []
    count = len(coordinates)
    if count < 10:
        warnings.append(f'{count} stations: narrow-lane FCBs from fewer than 10 stations are weak (Li & Zhang 2012 use '
                        '>= 10 regional stations; global products use 80-200).')
    if user_xyz is not None and count:
        distance = np.linalg.norm(np.asarray(list(coordinates.values()))-np.asarray(user_xyz, float), axis=1)/1e3
        near = int((distance <= 2000).sum())
        if near < min(10, count):
            warnings.append(f'Only {near} of {count} stations are within 2000 km of the user station '
                            f'(median {np.median(distance):.0f} km); regional FCBs should come from nearby stations.')
    return warnings


def drop_sparse(result, min_contributors):
    """Remove satellite UFOs of blocks observed by fewer than ``min_contributors`` stations."""
    if min_contributors <= 1 or result.rows.empty:
        return 0
    counts = result.rows[result.rows.fixed].groupby(['block', 'sat']).station.nunique()
    keys = list(zip(result.satellite.block, result.satellite.sat))
    keep = np.array([counts.get(key, 0) >= min_contributors for key in keys], bool)
    dropped = int((~keep).sum())
    result.satellite = result.satellite[keep].reset_index(drop=True)
    return dropped


def wide_lane_rows(name, ambiguities, min_epochs, wl_interval):
    rows = []
    for (sat, arc), g in ambiguities.dropna(subset=['A4']).groupby(['sat', 'arc']):
        if len(g) < min_epochs:
            continue
        cycles = g.A4.to_numpy(float)/gnss.constants(gnss.system_of(sat))['lambda_w']
        middle = float(g.epok.median())
        rows.append(dict(station=name, sat=int(sat), arc=int(arc), block=int(middle//wl_interval), epok=float(middle//wl_interval*wl_interval),
                         value=float(cycles.mean()), sigma=max(float(cycles.std(ddof=1)/np.sqrt(len(cycles))), 0.02),
                         elevation=float(g.elevation.mean()), arc_epochs=len(g)))
    return rows


def narrow_lane_rows(name, ambiguities, wide_lanes, interval, max_sigma, min_epochs):
    """``wide_lanes``: {(sat, arc): integer WL}. Float IF ambiguity at the end of each block."""
    rows = []
    for (sat, arc), g in ambiguities.groupby(['sat', 'arc']):
        if (sat, arc) not in wide_lanes:
            continue
        g = g.sort_values('epok').assign(arc_epochs=np.arange(1, len(g)+1))
        g = g[g.arc_epochs >= min_epochs]
        k = gnss.constants(gnss.system_of(sat))
        for block, b in g.groupby(g.epok//interval):
            last = b.iloc[-1]
            sigma = float(last.sigma_N3)/k['lambda_n']
            if sigma > max_sigma:
                continue
            rows.append(dict(station=name, sat=int(sat), arc=int(arc), block=int(block), epok=float(block*interval),
                             value=(float(last.N3)-k['beta']*wide_lanes[sat, arc])/k['lambda_n'], sigma=sigma,
                             elevation=float(last.elevation), arc_epochs=int(last.arc_epochs)))
    return rows


def integer_wide_lanes(result, rows, gate):
    """{(station, sat, arc): integer WL} for arcs whose corrected fraction passes the gate."""
    out = {}
    receiver = {(r.station, r.block): r.ufo for r in result.receiver.itertuples()}
    for r in rows.itertuples():
        s = result.lookup(r.sat, np.array([r.epok]))[0]
        offset = receiver.get((r.station, r.block))
        if not np.isfinite(s) or offset is None:
            continue
        corrected = r.value-offset-s
        if abs(corrected-np.rint(corrected)) < gate:
            out[r.station, r.sat, r.arc] = int(np.rint(corrected))
    return out


def l1_l2(nl, wl, system='G'):
    """GENFCB FCBs (cycles) of the two carriers (GPS L1/L2, Galileo E1/E5a) from NL and WL UFOs."""
    k = gnss.constants(system)
    b1 = nl-k['beta']/k['lambda_n']*wl
    return b1, b1-wl


def write_fcb(path, parts, date, interval, comment):
    """``parts``: {system: (nl, wl)}. GPS second carrier in the L2 column, Galileo E5a in L5."""
    lines = ['## FRACTIONAL CYCLE BIAS BY TOGU GNSS-PROCESSING FCB ESTIMATION', '#s 3 L1 L2 L5',
             f'# {comment}', '# UNIT: CYCLE',
             '#      ARC-START           ARC-END       SAT    L1-FCB  L1-STD    L2-FCB  L2-STD    L5-FCB  L5-STD']
    origin = pd.Timestamp(date)
    records = []
    for system, (nl, wl) in parts.items():
        k = gnss.constants(system)
        for r in nl.satellite.itertuples():
            w = wl.lookup(r.sat, np.array([r.epok]))[0]
            if not np.isfinite(w):
                continue
            sigma_w = float(wl.satellite[wl.satellite.sat == r.sat].sigma.iloc[-1])
            b1, b2 = l1_l2(r.ufo, w, system)
            s1 = float(np.hypot(r.sigma, k['beta']/k['lambda_n']*sigma_w)) if np.isfinite(r.sigma) else 0.0
            s2 = float(np.hypot(s1, sigma_w)) if np.isfinite(sigma_w) else s1
            second = (f'{b2:9.5f} {s2:7.5f}   0.00000 0.00000' if system == 'G' else f'{0:9.5f} {0:7.5f} {b2:9.5f} {s2:7.5f}')
            records.append((r.epok, r.sat, b1, s1, second))
    for epok, sat, b1, s1, second in sorted(records, key=lambda x: (x[0], x[1])):
        start, end = origin+pd.Timedelta(seconds=epok), origin+pd.Timedelta(seconds=epok+interval)
        lines.append(f'{start:%Y/%m/%d %H:%M:%S} {end:%Y/%m/%d %H:%M:%S}  {gnss.label(sat)} {b1:9.5f} {s1:7.5f} {second}')
    Path(path).write_text('\n'.join(lines)+'\n')


def estimate_fcb(config, stations, methods=core.METHODS, interval=900.0, wl_interval=86400.0, gate=0.25,
                 max_nl_sigma=0.2, min_epochs=20, validate=True, progress=None):
    """Run the whole workflow; writes <output>/fcb/ and returns a dict with the results."""
    progress = progress or (lambda p, m: None)
    paths = validate_stations(stations)
    for method in methods:
        if method not in core.METHODS:
            raise ValueError(f'FCB method must be one of {", ".join(core.METHODS)}.')
    output = Path(config.output)/'fcb'
    output.mkdir(parents=True, exist_ok=True)
    data, names, date, coordinates = [], set(), None, {}
    for k, (path, (name, day, ambiguities, xyz)) in enumerate(zip(paths, process_stations(config, paths, progress,
                                                                                           getattr(config, 'fcb_workers', 0)))):
        if date is not None and day.date() != date.date():
            raise ValueError(f'{path.name}: observation day differs from the other stations.')
        if name in names:
            name = f'{name}_{k}'
        names.add(name)
        date = day
        data.append((name, ambiguities))
        coordinates[name] = xyz
    wl_all = pd.DataFrame([row for name, a in data for row in wide_lane_rows(name, a, min_epochs, wl_interval)])
    wl_all.to_csv(output/'wide_lane_ambiguities.csv', index=False)
    systems = list(getattr(config, 'systems', ('G',)))
    user_xyz = None
    if getattr(config, 'rinex', '') and Path(str(config.rinex)).name not in {p.name for p in paths}:
        try:
            from gps_processing.io.rinex import read_header
            from gps_processing.io.ionex import read_text
            user_xyz = read_header(read_text(config.rinex)[:200])[2]
        except (OSError, ValueError, IndexError):
            user_xyz = None
    summary, results = {'stations': sorted(names), 'interval_s': interval, 'methods': {}, 'systems': {},
                        'hold_coordinates': bool(getattr(config, 'fcb_hold_coordinates', True)),
                        'warnings': network_warnings(coordinates, user_xyz)}, {}
    # 0 = automatic: one per ten stations (5 stations: no filter; 100 stations: 10 as in Geng et al. 2012).
    min_contributors = int(getattr(config, 'fcb_min_contributors', 0)) or int(np.clip(len(names)//10, 1, 10))
    summary['min_contributors'] = min_contributors
    usable = []
    for system in systems:
        table = wl_all[gnss.systems_of(wl_all.sat.to_numpy()) == system] if len(wl_all) else wl_all
        count = table.station.nunique() if len(table) else 0
        summary['systems'][system] = dict(stations=int(count), used=count >= MIN_STATIONS)
        if count >= MIN_STATIONS:
            usable.append(system)
    if not usable:
        raise ValueError('FCB estimation needs at least 5 stations tracking each system: '
                         + ', '.join(f'{gnss.NAMES[s]} {summary["systems"][s]["stations"]}' for s in systems))
    for k, method in enumerate(methods):
        progress(60+int(35*k/len(methods)), f'FCB estimation: {method}')
        slug = method.replace('+', '_')
        parts, entry, satellites_out, receivers_out, loo_all = {}, {}, [], [], []
        for system in usable:
            wl_table = wl_all[gnss.systems_of(wl_all.sat.to_numpy()) == system]
            wl = core.estimate(wl_table, method, gate)
            wide = integer_wide_lanes(wl, wl_table, gate)
            nl_rows = [row for name, a in data
                       for row in narrow_lane_rows(name, a[gnss.systems_of(a.sat.to_numpy()) == system],
                                                   {(s, arc): n for (st, s, arc), n in wide.items() if st == name},
                                                   interval, max_nl_sigma, min_epochs)]
            nl_table = pd.DataFrame(nl_rows)
            nl_table.to_csv(output/f'narrow_lane_ambiguities_{slug}_{system}.csv', index=False)
            if nl_table.empty or nl_table.station.nunique() < 2:
                entry[system] = dict(wl_arcs=len(wl_table), wl_fixed=len(wide), error='no converged narrow-lane ambiguities')
                continue
            nl = core.estimate(nl_table, method, gate)
            sparse = drop_sparse(nl, min_contributors)
            parts[system] = (nl, wl)
            satellites_out += [nl.satellite.assign(lane='NL', system=system), wl.satellite.assign(lane='WL', system=system)]
            receivers_out += [nl.receiver.assign(lane='NL', system=system), wl.receiver.assign(lane='WL', system=system)]
            part = dict(wl_arcs=len(wl_table), wl_fixed=len(wide), nl_ambiguities=len(nl_table),
                        nl_per_station={str(key): int(v) for key, v in nl_table.station.value_counts().items()},
                        nl_fixed_fraction=float(nl.rows.fixed.mean()) if len(nl.rows) else 0.0,
                        nl_residual_rms=float(np.sqrt(np.mean(nl.rows.residual[nl.rows.fixed]**2))) if nl.rows.fixed.any() else np.nan,
                        datum=nl.datum, sparse_fcbs_dropped=sparse, **nl.info)
            if validate:
                # Wide lane: one block per day, so the receiver part is removed over the whole day.
                wl_loo = core.leave_one_out(wl_table.assign(block=0, epok=0.0), method, gate)
                if len(wl_loo):
                    part.update(wl_loo_rms_cycles=float(wl_loo.rms_cycles.mean()), wl_loo_within_015=float(wl_loo.within_015.mean()))
                loo = core.leave_one_out(nl_table, method, gate)
                if len(loo):
                    loo_all.append(loo.assign(system=system))
                    part.update(loo_rms_cycles=float(loo.rms_cycles.mean()), loo_within_015=float(loo.within_015.mean()),
                                loo_within_025=float(loo.within_025.mean()))
            entry[system] = part
            results.setdefault(method, {})[system] = dict(wl=wl, nl=nl, nl_table=nl_table, wl_table=wl_table)
        if not parts:
            raise ValueError('No converged narrow-lane ambiguities (float sigma below the limit); use longer observation files.')
        write_fcb(output/f'fcb_{slug}.fcb', parts, date, interval,
                  f'method {method}; orbit {Path(config.sp3_day).name}; clock {Path(config.clock_file).name if config.clock_file else "SP3"}; '
                  f'systems {",".join(parts)}; stations {",".join(sorted(names))}')
        pd.concat(satellites_out).to_csv(output/f'satellite_ufo_{slug}.csv', index=False)
        pd.concat(receivers_out).to_csv(output/f'receiver_ufo_{slug}.csv', index=False)
        if loo_all:
            loo = pd.concat(loo_all)
            loo.to_csv(output/f'validation_{slug}.csv', index=False)
            entry.update(loo_rms_cycles=float(np.average(loo.rms_cycles, weights=loo.ambiguities)),
                         loo_within_015=float(np.average(loo.within_015, weights=loo.ambiguities)),
                         loo_within_025=float(np.average(loo.within_025, weights=loo.ambiguities)))
        summary['methods'][method] = entry
    scored = {m: e['loo_rms_cycles'] for m, e in summary['methods'].items() if 'loo_rms_cycles' in e}
    summary['best_method'] = min(scored, key=scored.get) if scored else None
    (output/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    progress(100, f'FCB estimation finished: {output}')
    return dict(summary=summary, results=results, output=output)
