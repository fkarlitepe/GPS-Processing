"""FCB estimation from a reference network (at least five stations), Li & Zhang (2012) workflow.

1. Every station is processed with the float EKF PPP (same orbit, clock, ANTEX and models the
   users will apply); float zero-difference ambiguities are kept with their sigma and arc.
2. Wide lane: arc means of the MW combination (cycles) -> WL UFOs (one block per day by
   default) -> integer WL ambiguity of every arc whose fractional residual passes the gate.
3. Narrow lane: B_NL = (B_IF - beta*N_WL)/lambda_NL from the float IF ambiguity at the end of
   each block (default 15 min) -> NL UFOs per block.
4. The UFOs are written as GENFCB L1/L2 FCBs, b1 = s_NL - (beta/lambda_NL)*s_WL and
   b2 = b1 - s_WL, which io.fcb_products turns back into the same WL and IF corrections.
Each estimator (batch / EKF / EKF+ML) is validated by leaving each station out in turn.
"""
from dataclasses import replace, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gps_processing.processing.ar.models import BETA, LAMBDA_N, LAMBDA_W
from gps_processing.processing.fcb import core

MIN_STATIONS = 5


def validate_stations(stations):
    from gps_processing.data_directory import require_data_file
    paths = [require_data_file(p) for p in stations]
    if len(set(paths)) < MIN_STATIONS:
        raise ValueError(f'FCB kestirimi için en az {MIN_STATIONS} farklı istasyon RINEX dosyası gerekli ({len(set(paths))} verildi).')
    return paths


def station_ambiguities(config, path, progress=lambda p, m: None):
    """Float PPP of one station -> (name, date, MW per epoch, float IF ambiguities per epoch)."""
    from gps_processing.pipeline import prepare, antenna_models
    from gps_processing.processing import ppp_ekf
    from gps_processing.processing.atmosphere import Atmosphere
    station = replace(config, rinex=path, technique='PPP', estimator='ekf' if config.estimator == 'sequential' else config.estimator,
                      ml_weighting=False, troposphere='')
    orbit, obs, satellites = prepare(station, progress)
    obs, satellites, _ = antenna_models(station, obs, satellites)
    atmosphere = Atmosphere.from_config(station, obs)
    output = ppp_ekf.solve(obs.combined, satellites, obs.approximate_xyz, station, atmosphere=atmosphere)
    name = (obs.marker or Path(path).name)[:9].strip() or Path(path).stem[:4]
    ambiguities = output.ambiguities.merge(obs.combined[['epok', 'sat', 'A4']], on=['epok', 'sat'], how='left')
    return name, obs.date, ambiguities


def wide_lane_rows(name, ambiguities, min_epochs, wl_interval):
    rows = []
    for (sat, arc), g in ambiguities.dropna(subset=['A4']).groupby(['sat', 'arc']):
        if len(g) < min_epochs:
            continue
        cycles = g.A4.to_numpy(float)/LAMBDA_W
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
        for block, b in g.groupby(g.epok//interval):
            last = b.iloc[-1]
            sigma = float(last.sigma_N3)/LAMBDA_N
            if sigma > max_sigma:
                continue
            rows.append(dict(station=name, sat=int(sat), arc=int(arc), block=int(block), epok=float(block*interval),
                             value=(float(last.N3)-BETA*wide_lanes[sat, arc])/LAMBDA_N, sigma=sigma,
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


def l1_l2(nl, wl):
    """GENFCB L1/L2 FCBs (cycles) from narrow- and wide-lane satellite UFOs."""
    b1 = nl-BETA/LAMBDA_N*wl
    return b1, b1-wl


def write_fcb(path, nl, wl, date, interval, comment):
    lines = ['## FRACTIONAL CYCLE BIAS BY TOGU GPS-PROCESSING FCB ESTIMATION', '#s 3 L1 L2 L5',
             f'# {comment}', '# UNIT: CYCLE',
             '#      ARC-START           ARC-END       SAT    L1-FCB  L1-STD    L2-FCB  L2-STD    L5-FCB  L5-STD']
    origin = pd.Timestamp(date)
    for r in nl.satellite.sort_values(['epok', 'sat']).itertuples():
        w = wl.lookup(r.sat, np.array([r.epok]))[0]
        if not np.isfinite(w):
            continue
        sigma_w = float(wl.satellite[wl.satellite.sat == r.sat].sigma.iloc[-1])
        b1, b2 = l1_l2(r.ufo, w)
        s1 = float(np.hypot(r.sigma, BETA/LAMBDA_N*sigma_w)) if np.isfinite(r.sigma) else 0.0
        s2 = float(np.hypot(s1, sigma_w)) if np.isfinite(sigma_w) else s1
        start, end = origin+pd.Timedelta(seconds=r.epok), origin+pd.Timedelta(seconds=r.epok+interval)
        lines.append(f'{start:%Y/%m/%d %H:%M:%S} {end:%Y/%m/%d %H:%M:%S}  G{r.sat:02d} {b1:9.5f} {s1:7.5f} {b2:9.5f} {s2:7.5f}   0.00000 0.00000')
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
    data, names, date = [], set(), None
    for k, path in enumerate(paths):
        progress(5+int(55*k/len(paths)), f'Float PPP for FCB: station {k+1}/{len(paths)} ({path.name})')
        name, day, ambiguities = station_ambiguities(config, path)
        if date is not None and day.date() != date.date():
            raise ValueError(f'{path.name}: observation day differs from the other stations.')
        if name in names:
            name = f'{name}_{k}'
        names.add(name)
        date = day
        data.append((name, ambiguities))
    wl_table = pd.DataFrame([row for name, a in data for row in wide_lane_rows(name, a, min_epochs, wl_interval)])
    wl_table.to_csv(output/'wide_lane_ambiguities.csv', index=False)
    if wl_table.empty or wl_table.station.nunique() < 2:
        raise ValueError('Too few wide-lane arcs: stations need longer common observation arcs.')
    summary, results = {'stations': sorted(names), 'interval_s': interval, 'methods': {}}, {}
    for k, method in enumerate(methods):
        progress(60+int(35*k/len(methods)), f'FCB estimation: {method}')
        wl = core.estimate(wl_table, method, gate)
        wide = integer_wide_lanes(wl, wl_table, gate)
        nl_rows = [row for name, a in data
                   for row in narrow_lane_rows(name, a, {(s, arc): n for (st, s, arc), n in wide.items() if st == name},
                                               interval, max_nl_sigma, min_epochs)]
        nl_table = pd.DataFrame(nl_rows)
        nl_table.to_csv(output/f'narrow_lane_ambiguities_{method.replace("+", "_")}.csv', index=False)
        if nl_table.empty or nl_table.station.nunique() < 2:
            raise ValueError('No converged narrow-lane ambiguities (float sigma below the limit); use longer observation files.')
        nl = core.estimate(nl_table, method, gate)
        slug = method.replace('+', '_')
        write_fcb(output/f'fcb_{slug}.fcb', nl, wl, date, interval,
                  f'method {method}; orbit {Path(config.sp3_day).name}; clock {Path(config.clock_file).name if config.clock_file else "SP3"}; stations {",".join(sorted(names))}')
        nl.satellite.assign(lane='NL').pipe(lambda f: pd.concat([f, wl.satellite.assign(lane='WL')])).to_csv(output/f'satellite_ufo_{slug}.csv', index=False)
        pd.concat([nl.receiver.assign(lane='NL'), wl.receiver.assign(lane='WL')]).to_csv(output/f'receiver_ufo_{slug}.csv', index=False)
        entry = dict(wl_arcs=len(wl_table), wl_fixed=len(wide), nl_ambiguities=len(nl_table),
                     nl_per_station={str(k): int(v) for k, v in nl_table.station.value_counts().items()},
                     nl_fixed_fraction=float(nl.rows.fixed.mean()) if len(nl.rows) else 0.0,
                     nl_residual_rms=float(np.sqrt(np.mean(nl.rows.residual[nl.rows.fixed]**2))) if nl.rows.fixed.any() else np.nan,
                     datum=nl.datum, **nl.info)
        if validate:
            # Wide lane: one block per day, so the receiver part is removed over the whole day.
            wl_loo = core.leave_one_out(wl_table.assign(block=0, epok=0.0), method, gate)
            if len(wl_loo):
                entry.update(wl_loo_rms_cycles=float(wl_loo.rms_cycles.mean()), wl_loo_within_015=float(wl_loo.within_015.mean()))
            loo = core.leave_one_out(nl_table, method, gate)
            loo.to_csv(output/f'validation_{slug}.csv', index=False)
            if len(loo):
                entry.update(loo_rms_cycles=float(loo.rms_cycles.mean()), loo_within_015=float(loo.within_015.mean()),
                             loo_within_025=float(loo.within_025.mean()))
        summary['methods'][method] = entry
        results[method] = dict(wl=wl, nl=nl, nl_table=nl_table, wl_table=wl_table)
    scored = {m: e['loo_rms_cycles'] for m, e in summary['methods'].items() if 'loo_rms_cycles' in e}
    summary['best_method'] = min(scored, key=scored.get) if scored else None
    (output/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    progress(100, f'FCB estimation finished: {output}')
    return dict(summary=summary, results=results, output=output)
