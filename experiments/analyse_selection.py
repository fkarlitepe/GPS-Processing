"""Tables and figures of experiment S (experiments/run_selection.py): satellite selection in float PPP,
classical DOP vs weighted DOP (WDOP).

    python3 experiments/analyse_selection.py      -> results/experiments/selection/summary.md, fig_*.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'results'/'experiments'/'selection'
WEIGHT_NAMES = {'none': 'none (DOP)', 'elevation': 'sin²el (WDOP)', 'cn0': 'C/N0 (WDOP)', 'filter': 'filter σ (WDOP)'}
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5cc4']


def load():
    s = pd.read_csv(D/'sessions.csv')
    if 'error' in s:
        s = s[s.error.isna()].drop(columns='error')
    s['rms_3d'] = np.sqrt(s.rms_e**2+s.rms_n**2+s.rms_u**2)
    s['rms_h'] = np.hypot(s.rms_e, s.rms_n)
    s['converged'] = np.isfinite(s.convergence_min)
    # Unconverged sessions count as 120 min (session length) in the convergence statistics.
    s['conv'] = s.convergence_min.fillna(120.0)
    return s


def fmt(v, d=2):
    return '—' if v is None or not np.isfinite(v) else f'{v:.{d}f}'


def paired(a, b, key, column):
    """Wilcoxon signed-rank on paired station-session values; returns (median diff, % a better, p)."""
    x = a.set_index(key)[column]
    y = b.set_index(key)[column]
    j = pd.concat([x, y], axis=1, keys=['a', 'b']).dropna()
    if len(j) < 8:
        return np.nan, np.nan, np.nan, len(j)
    d = j.a-j.b
    nz = d[np.abs(d) > 1e-12]
    p = stats.wilcoxon(nz).pvalue if len(nz) > 8 else np.nan
    return float(d.median()), float(100*(d < 0).mean()), p, len(j)


def summary_table(s, out):
    out.append('| k | setting | sessions | converged % | conv. median (min) | RMS E (mm) | RMS N (mm) | RMS U (mm) | RMS 3D (mm) | '
               'median PDOP | median WPDOP | re-selections / session | ms/epoch |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    order = ['ALL'] + sorted((x for x in s.setting.unique() if x != 'ALL'), key=lambda x: (x.split('|')[0], x))
    for k in sorted(s.k.unique()):
        for setting in order:
            g = s[(s.k == k) & (s.setting == setting)]
            if g.empty:
                continue
            out.append(f'| {k if k else "all"} | {setting} | {len(g)} | {100*g.converged.mean():.0f} | {fmt(g.conv.median(), 1)} | '
                       f'{fmt(1e3*g.rms_e.median(), 1)} | {fmt(1e3*g.rms_n.median(), 1)} | {fmt(1e3*g.rms_u.median(), 1)} | '
                       f'{fmt(1e3*g.rms_3d.median(), 1)} | {fmt(g.median_pdop.median())} | {fmt(g.median_wpdop.median())} | '
                       f'{fmt(g.reselections.mean(), 1)} | {fmt(g.selection_ms.mean(), 1)} |')


def weighting_table(s, out):
    key = ['station', 'session']
    out.append('| k | method | weighting | conv. median (min) | RMS 3D median (mm) | Δconv vs none (min) | better % | p | '
               'ΔRMS 3D vs none (mm) | better % | p |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for k in sorted(s.k.unique()):
        if k == 0:
            continue
        for method in ('FIBO-GNSS+LS', 'BACKWARD+LS'):
            base = s[(s.k == k) & (s.method == method) & (s.weighting == 'none') & s.hysteresis]
            for w in ('none', 'elevation', 'cn0', 'filter'):
                g = s[(s.k == k) & (s.method == method) & (s.weighting == w) & s.hysteresis]
                if g.empty:
                    continue
                if w == 'none':
                    out.append(f'| {k} | {method} | {WEIGHT_NAMES[w]} | {fmt(g.conv.median(), 1)} | {fmt(1e3*g.rms_3d.median(), 1)} | — | — | — | — | — | — |')
                    continue
                dc, bc, pc, _ = paired(g, base, key, 'conv')
                dr, br, pr, _ = paired(g, base, key, 'rms_3d')
                out.append(f'| {k} | {method} | {WEIGHT_NAMES[w]} | {fmt(g.conv.median(), 1)} | {fmt(1e3*g.rms_3d.median(), 1)} | '
                           f'{fmt(dc, 1)} | {fmt(bc, 0)} | {fmt(pc, 3)} | {fmt(1e3*dr, 1)} | {fmt(br, 0)} | {fmt(pr, 3)} |')


def method_table(s, out):
    key = ['station', 'session']
    out.append('| k | weighting | comparison | Δconv (min) | A better % | p | ΔRMS 3D (mm) | A better % | p | n |')
    out.append('|---|---|---|---|---|---|---|---|---|---|')
    for k in sorted(s.k.unique()):
        if k == 0:
            continue
        for w in ('none', 'filter'):
            a = s[(s.k == k) & (s.method == 'FIBO-GNSS+LS') & (s.weighting == w) & s.hysteresis]
            for other, ow in (('BACKWARD+LS', w), ('TOP-CN0', 'cn0'), ('RANDOM', 'none')):
                b = s[(s.k == k) & (s.method == other) & (s.weighting == ow)]
                dc, bc, pc, n = paired(a, b, key, 'conv')
                dr, br, pr, _ = paired(a, b, key, 'rms_3d')
                out.append(f'| {k} | {w} | FIBO-GNSS+LS vs {other} | {fmt(dc, 1)} | {fmt(bc, 0)} | {fmt(pc, 3)} | '
                           f'{fmt(1e3*dr, 1)} | {fmt(br, 0)} | {fmt(pr, 3)} | {n} |')
    allv = s[s.setting == 'ALL']
    for k in sorted(s.k.unique()):
        if k == 0:
            continue
        a = s[(s.k == k) & (s.method == 'FIBO-GNSS+LS') & (s.weighting == 'filter') & s.hysteresis]
        dc, bc, pc, n = paired(a, allv, key, 'conv')
        dr, br, pr, _ = paired(a, allv, key, 'rms_3d')
        out.append(f'| {k} | filter | FIBO-GNSS+LS vs ALL visible | {fmt(dc, 1)} | {fmt(bc, 0)} | {fmt(pc, 3)} | {fmt(1e3*dr, 1)} | {fmt(br, 0)} | {fmt(pr, 3)} | {n} |')


def dop_vs_accuracy(s, out):
    out.append('| k | Spearman ρ (session median PDOP, conv.) | ρ (median WPDOP, conv.) | ρ (median PDOP, RMS 3D) | ρ (median WPDOP, RMS 3D) | ρ (median PDOP, σU) | ρ (median WPDOP, σU) |')
    out.append('|---|---|---|---|---|---|---|')
    for k in sorted(s.k.unique()):
        if k == 0:
            continue
        g = s[(s.k == k) & s.method.isin(['FIBO-GNSS+LS', 'BACKWARD+LS', 'TOP-CN0', 'RANDOM'])]
        r = lambda a, b: stats.spearmanr(g[a], g[b], nan_policy='omit').statistic
        out.append(f'| {k} | {fmt(r("median_pdop", "conv"))} | {fmt(r("median_wpdop", "conv"))} | {fmt(r("median_pdop", "rms_3d"))} | '
                   f'{fmt(r("median_wpdop", "rms_3d"))} | {fmt(r("median_pdop", "mean_sigma_u"))} | {fmt(r("median_wpdop", "mean_sigma_u"))} |')


def hysteresis_table(s, out):
    key = ['station', 'session']
    a = s[(s.k == 8) & (s.method == 'FIBO-GNSS+LS') & (s.weighting == 'filter') & s.hysteresis]
    b = s[(s.k == 8) & (s.setting == 'FIBO-GNSS+LS|filter|no-hysteresis')]
    if a.empty or b.empty:
        return
    dc, bc, pc, n = paired(a, b, key, 'conv')
    dr, br, pr, _ = paired(a, b, key, 'rms_3d')
    out.append('| k = 8, FIBO-GNSS+LS, filter | re-selections / session | conv. median (min) | RMS 3D median (mm) |')
    out.append('|---|---|---|---|')
    for name, g in (('hysteresis 5 %', a), ('every epoch', b)):
        out.append(f'| {name} | {fmt(g.reselections.mean(), 1)} | {fmt(g.conv.median(), 1)} | {fmt(1e3*g.rms_3d.median(), 1)} |')
    out.append(f'\nPaired (hysteresis − every epoch): Δconv {fmt(dc, 1)} min (better in {fmt(bc, 0)} %, p = {fmt(pc, 3)}), '
               f'ΔRMS 3D {fmt(1e3*dr, 1)} mm (better in {fmt(br, 0)} %, p = {fmt(pr, 3)}), n = {n}.')


def svg_lines(fname, title, ylabel, xs, series, ref=None, ref_label=''):
    W, H, L, R, T, B = 760, 420, 64, 220, 56, 50
    vals = [v for _, ys in series for v in ys if np.isfinite(v)] + ([ref] if ref is not None else [])
    lo, hi = 0.0, max(vals)*1.1
    X = lambda i: L+i/(len(xs)-1)*(W-L-R)
    Y = lambda v: T+(hi-v)/(hi-lo)*(H-T-B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>', f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="#1a1a19">{title}</text>',
         f'<text x="{L}" y="42" fill="#5f5e5a">float PPP G+E, 2-h sessions, median over stations and sessions</text>']
    for t in np.linspace(lo, hi, 5):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#e6e5e0"/>')
        o.append(f'<text x="{L-8}" y="{Y(t)+4:.1f}" text-anchor="end" fill="#5f5e5a">{t:.1f}</text>')
    for i, k in enumerate(xs):
        o.append(f'<text x="{X(i):.1f}" y="{H-B+20}" text-anchor="middle" fill="#5f5e5a">{k}</text>')
    o.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-12}" text-anchor="middle" fill="#5f5e5a">satellites per epoch k</text>')
    o.append(f'<text x="16" y="{(T+H-B)/2:.0f}" transform="rotate(-90 16 {(T+H-B)/2:.0f})" text-anchor="middle" fill="#5f5e5a">{ylabel}</text>')
    if ref is not None:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" stroke="#5f5e5a" stroke-dasharray="5 4"/>')
        o.append(f'<text x="{W-R-4}" y="{Y(ref)-6:.1f}" text-anchor="end" fill="#5f5e5a">{ref_label}</text>')
    for i, (label, ys) in enumerate(series):
        c = PAL[i % len(PAL)]
        pts = ' '.join(f'{X(j):.1f},{Y(v):.1f}' for j, v in enumerate(ys) if np.isfinite(v))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"><title>{label}</title></polyline>')
        o += [f'<circle cx="{X(j):.1f}" cy="{Y(v):.1f}" r="4" fill="{c}" stroke="#fcfcfb" stroke-width="1.5"><title>{label}, k={xs[j]}: {v:.2f}</title></circle>'
              for j, v in enumerate(ys) if np.isfinite(v)]
        o.append(f'<circle cx="{W-R+22}" cy="{T+10+i*22}" r="4" fill="{c}"/><text x="{W-R+32}" y="{T+14+i*22}" fill="#1a1a19">{label}</text>')
    o.append('</svg>')
    (D/fname).write_text('\n'.join(o), encoding='utf-8')


def figures(s):
    ks = sorted(k for k in s.k.unique() if k)
    allv = s[s.setting == 'ALL']
    pick = [('FIBO-GNSS+LS', 'filter'), ('FIBO-GNSS+LS', 'none'), ('BACKWARD+LS', 'filter'), ('BACKWARD+LS', 'none'),
            ('TOP-CN0', 'cn0'), ('RANDOM', 'none')]
    def series(col, scale=1.0):
        return [(f'{m} · {WEIGHT_NAMES[w]}', [scale*s[(s.k == k) & (s.method == m) & (s.weighting == w) & (s.hysteresis | (s.method == 'RANDOM'))][col].median() for k in ks])
                for m, w in pick]
    svg_lines('fig_convergence_vs_k.svg', 'Convergence time (10 cm H and V, 20 epochs)', 'median convergence (min)', ks,
              series('conv'), allv.conv.median(), 'all visible')
    svg_lines('fig_rms3d_vs_k.svg', 'Final accuracy (last 30 min of each session)', 'median RMS 3D (mm)', ks,
              series('rms_3d', 1e3), 1e3*allv.rms_3d.median(), 'all visible')
    svg_lines('fig_wpdop_vs_k.svg', 'Median WPDOP (filter weights) of the used satellites', 'median WPDOP', ks,
              series('median_wpdop'), allv.median_wpdop.median(), 'all visible')


def main():
    s = load()
    out = ['# Experiment S: satellite selection in float PPP — DOP vs weighted DOP\n',
           f'{s.station.nunique()} stations ({", ".join(sorted(s.station.unique()))}), 2023-11-23, 2-h sessions, GPS + Galileo, '
           'COD MGEX final products, static EKF. Convergence: horizontal and vertical error ≤ 10 cm for 20 epochs (unconverged '
           'sessions count as 120 min). Accuracy: RMS over the last 30 min of each session. Medians over station-sessions.\n',
           '## 1. All settings\n']
    summary_table(s, out)
    out += ['\n## 2. Effect of the DOP weighting (WDOP vs classical DOP) on PPP\n',
            'Paired by station and session against the unweighted selection of the same method and k; negative Δ = the weighted '
            'selection is better; p: Wilcoxon signed-rank.\n']
    weighting_table(s, out)
    out += ['\n## 3. Selection methods\n']
    method_table(s, out)
    out += ['\n## 4. Does the DOP of the used subset predict the PPP result?\n']
    dop_vs_accuracy(s, out)
    out += ['\n## 5. Hysteresis (keeping phase arcs)\n']
    hysteresis_table(s, out)
    figures(s)
    out.append('\nFigures: `fig_convergence_vs_k.svg`, `fig_rms3d_vs_k.svg`, `fig_wpdop_vs_k.svg`.')
    (D/'summary.md').write_text('\n'.join(out)+'\n', encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
