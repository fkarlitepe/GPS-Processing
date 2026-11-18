"""Solution quality: local ENU errors, convergence time and method statistics."""
import numpy as np
import pandas as pd
from gps_processing.processing.troposphere import geodetic


def enu_rotation(xyz):
    lat, lon, _ = geodetic(xyz)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl*co, -sl*so, cl], [cl*co, cl*so, sl]])


def add_enu(positions, reference):
    """Append east/north/up and horizontal/3D errors (m) relative to ``reference``."""
    reference = np.asarray(reference, dtype=float)
    delta = positions[['x', 'y', 'z']].to_numpy(dtype=float)-reference
    enu = delta@enu_rotation(reference).T
    frame = positions.copy()
    frame['east'], frame['north'], frame['up'] = enu.T
    frame['error_h'] = np.hypot(enu[:, 0], enu[:, 1])
    frame['error_3d'] = np.linalg.norm(enu, axis=1)
    return frame


def convergence_time(frame, horizontal=0.10, vertical=0.20):
    """Seconds from the first epoch until errors stay below the thresholds; NaN if never."""
    ok = (frame.error_h <= horizontal) & (frame.up.abs() <= vertical)
    bad = np.flatnonzero(~ok.to_numpy())
    if len(bad) == len(frame):
        return np.nan
    start = 0 if len(bad) == 0 else bad[-1]+1
    if start >= len(frame):
        return np.nan
    return float(frame.epok.iloc[start]-frame.epok.iloc[0])


def statistics(positions, reference, label='', reference_source='', horizontal=0.10, vertical=0.20):
    frame = add_enu(positions, reference)
    half = frame.iloc[len(frame)//2:]
    fixed = frame.status.isin(['fixed', 'partial_fixed']).mean() if 'status' in frame else 0.0
    return dict(solution=label, epochs=len(frame), reference=reference_source,
                final_e=frame.east.iloc[-1], final_n=frame.north.iloc[-1], final_u=frame.up.iloc[-1],
                final_3d=frame.error_3d.iloc[-1],
                rms_e=float(np.sqrt(np.mean(half.east**2))), rms_n=float(np.sqrt(np.mean(half.north**2))),
                rms_u=float(np.sqrt(np.mean(half.up**2))), rms_3d=float(np.sqrt(np.mean(half.error_3d**2))),
                convergence_s=convergence_time(frame, horizontal, vertical), fix_rate=float(fixed),
                mean_satellites=float(frame.satellites.mean()),
                mean_nis=float(frame.nis.mean()) if 'nis' in frame else np.nan)


def statistics_table(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ satellite selection / DOP analysis
def selection_statistics(summary):
    """Selection columns of the method statistics (comparison table)."""
    if not summary:
        return {}
    keys = ('method', 'weighting', 'objective', 'mean_visible', 'mean_selected', 'reselections', 'mean_selection_ms',
            'median_pdop_all', 'median_pdop_selected', 'median_wpdop_all', 'median_wpdop_selected', 'median_objective_ratio')
    return {f'selection_{k}': summary.get(k) for k in keys}


PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')


def _svg_lines(series, title, subtitle, ylabel, width=760, height=320, ymax=None):
    """Minimal dependency-free SVG line chart: ``series`` = [(label, x (h), y)]."""
    L, R, T, B = 60, 170, 50, 40
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2][np.isfinite(s[2])] for s in series])
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs))
    y0 = 0.0
    y1 = float(ymax if ymax is not None else np.nanpercentile(ys, 99.5)*1.08) if len(ys) else 1.0
    X = lambda v: L+(v-x0)/max(x1-x0, 1e-9)*(width-L-R)
    Y = lambda v: T+(y1-np.clip(v, y0, y1))/max(y1-y0, 1e-9)*(height-T-B)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="#fcfcfb"/>',
           f'<text x="{L}" y="22" font-size="15" font-weight="600" fill="#1a1a19">{title}</text>',
           f'<text x="{L}" y="39" fill="#5f5e5a">{subtitle}</text>']
    for t in np.linspace(y0, y1, 5):
        out.append(f'<line x1="{L}" x2="{width-R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#e6e5e0"/>')
        out.append(f'<text x="{L-8}" y="{Y(t)+4:.1f}" text-anchor="end" fill="#5f5e5a">{t:.2f}</text>')
    for t in np.linspace(x0, x1, 7):
        out.append(f'<text x="{X(t):.1f}" y="{height-B+18}" text-anchor="middle" fill="#5f5e5a">{t:.0f}</text>')
    out.append(f'<text x="{(L+width-R)/2:.0f}" y="{height-6}" text-anchor="middle" fill="#5f5e5a">hours of day (GPS time)</text>')
    out.append(f'<text x="14" y="{(T+height-B)/2:.0f}" transform="rotate(-90 14 {(T+height-B)/2:.0f})" text-anchor="middle" fill="#5f5e5a">{ylabel}</text>')
    for i, (label, x, y) in enumerate(series):
        c = PALETTE[i % len(PALETTE)]
        pts = ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(x, y) if np.isfinite(b))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.5" stroke-linejoin="round"><title>{label}</title></polyline>')
        out.append(f'<line x1="{width-R+16}" x2="{width-R+36}" y1="{T+10+i*20}" y2="{T+10+i*20}" stroke="{c}" stroke-width="2.5"/>')
        out.append(f'<text x="{width-R+42}" y="{T+14+i*20}" fill="#1a1a19">{label}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def _svg_skyplot(sats, title, size=520):
    """Polar sky plot: every visible satellite position; selected (filled) vs excluded (grey)."""
    c, r0 = size/2, size/2-40
    pos = lambda el, az: (c+r0*(90-el)/90*np.sin(np.radians(az)), c+20-r0*(90-el)/90*np.cos(np.radians(az)))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size+20}" font-family="system-ui, sans-serif" font-size="11">',
           f'<rect width="{size}" height="{size+20}" fill="#fcfcfb"/>',
           f'<text x="12" y="20" font-size="14" font-weight="600" fill="#1a1a19">{title}</text>']
    for el in (0, 30, 60):
        out.append(f'<circle cx="{c}" cy="{c+20}" r="{r0*(90-el)/90:.1f}" fill="none" stroke="#e6e5e0"/>')
        out.append(f'<text x="{c+3}" y="{c+20-r0*(90-el)/90+12:.1f}" fill="#8a8984">{el}°</text>')
    for az, lab in ((0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')):
        x, y = pos(-6, az)
        out.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" fill="#5f5e5a">{lab}</text>')
    step = max(1, len(sats)//6000)
    thin = sats.iloc[::step]
    for sel, colour, radius in ((False, '#c9c8c2', 1.4), (True, None, 1.8)):
        part = thin[thin.selected.astype(bool) == sel]
        for system, col in (('G', '#2a78d6'), ('E', '#eb6834')):
            q = part[part.system == system]
            fill = colour or col
            out.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}"/>'
                       for x, y in zip(*pos(q.elevation.to_numpy(), q.azimuth.to_numpy())))
    out.append(f'<text x="12" y="{size+12}" fill="#5f5e5a">blue: GPS selected · orange: Galileo selected · grey: visible, not used</text>')
    out.append('</svg>')
    return '\n'.join(out)


def selection_report(result, directory):
    """Analysis section of a run: DOP / weighted-DOP time series, sky plot, per-satellite table (markdown)."""
    from pathlib import Path
    sel = result.selection
    epochs, sats, per, summary = sel['epochs'], sel['satellites'], sel['per_satellite'], sel['summary']
    directory = Path(directory)
    if epochs.empty:
        return None
    hours = epochs.epok.to_numpy(float)/3600
    w = summary['weighting']
    subtitle = f"method {summary['method']} · weights {w} · objective {summary['objective'].upper()}"
    (directory/'selection_dop.svg').write_text(_svg_lines(
        [('PDOP all-in-view', hours, epochs.pdop_all.to_numpy(float)), ('PDOP selected', hours, epochs.pdop_selected.to_numpy(float)),
         ('WPDOP all-in-view', hours, epochs.wpdop_all.to_numpy(float)), ('WPDOP selected', hours, epochs.wpdop_selected.to_numpy(float))],
        'Dilution of precision: all visible vs selected satellites', subtitle, 'DOP'), encoding='utf-8')
    (directory/'selection_counts.svg').write_text(_svg_lines(
        [('visible', hours, epochs.visible.to_numpy(float)), ('selected', hours, epochs.selected.to_numpy(float)),
         ('GPS selected', hours, epochs.selected_gps.to_numpy(float)), ('Galileo selected', hours, epochs.selected_galileo.to_numpy(float))],
        'Visible and selected satellites', subtitle, 'satellites', ymax=float(epochs.visible.max())+1), encoding='utf-8')
    if not sats.empty:
        (directory/'selection_skyplot.svg').write_text(_svg_skyplot(sats, f'Sky plot · {subtitle}'), encoding='utf-8')
    positions = result.positions
    lines = [f'# Satellite selection and DOP analysis\n', f'{subtitle}; {summary["epochs"]} epochs.\n',
             '| Quantity | All visible | Selected |', '|---|---|---|',
             f'| Mean number of satellites | {summary["mean_visible"]:.2f} | {summary["mean_selected"]:.2f} |']
    for name in ('gdop', 'pdop', 'hdop', 'vdop'):
        lines.append(f'| Median {name.upper()} | {summary[f"median_{name}_all"]:.3f} | {summary[f"median_{name}_selected"]:.3f} |')
        lines.append(f'| Median W{name.upper()} ({w}) | {summary[f"median_w{name}_all"]:.3f} | {summary[f"median_w{name}_selected"]:.3f} |')
    lines += ['', f'Re-selections: {summary["reselections"]}; mean selection time {summary["mean_selection_ms"]:.2f} ms/epoch.', '']
    if 'error_3d' in positions:
        half = positions.iloc[len(positions)//2:]
        lines.append(f'Position error (second half of the session, RMS): E {np.sqrt(np.mean(half.east**2))*1e3:.1f} mm, '
                     f'N {np.sqrt(np.mean(half.north**2))*1e3:.1f} mm, U {np.sqrt(np.mean(half.up**2))*1e3:.1f} mm.\n')
    lines += ['## Per satellite\n',
              '`selected_share`: share of its visible epochs in which the satellite was used. `median_correction_3d_mm`: median size of its '
              'own share of the Kalman position correction while used; `rms_code_residual_m`, `rms_phase_residual_mm`: post-fit residuals. `median_delta_dop_selected`: objective increase if it is removed; '
              '`median_delta_dop_excluded`: objective change if it had been added (negative = it would have helped).\n']
    if not per.empty:
        cols = [c for c in ('prn', 'visible_epochs', 'selected_share', 'mean_elevation', 'mean_cn0', 'mean_weight',
                            'median_correction_3d_mm', 'rms_code_residual_m', 'rms_phase_residual_mm', 'median_delta_dop_selected', 'median_delta_dop_excluded') if c in per]
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '---|'*len(cols))
        for _, row in per[cols].iterrows():
            lines.append('| ' + ' | '.join(str(row[c]) if c == 'prn' else (f'{row[c]:.3f}' if isinstance(row[c], float) else str(row[c])) for c in cols) + ' |')
    lines += ['', 'Figures: `selection_dop.svg`, `selection_counts.svg`, `selection_skyplot.svg`. Tables: `selection_epochs.csv`, '
              '`selection_satellites.csv` (every epoch and visible satellite: weight, selected, delta DOP, model terms, residuals, '
              'Kalman correction share), `selection_per_satellite.csv`.']
    path = directory/'selection_report.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return path
