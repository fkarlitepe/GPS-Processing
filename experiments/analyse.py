"""Statistics, tables and figures of the MADR experiments (reads results/experiments/*/metrics.csv
and positions/*.csv; independent of gps_processing so it runs in the plotting environment).

    ~/micromamba/envs/gmt/bin/python experiments/analyse.py products|ablation|network
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'experiments'
FIG = OUT/'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
COLORS = {'FCB-batch': '#1f77b4', 'FCB-ekf': '#ff7f0e', 'FCB-ekf_ml': '#2ca02c', 'PPP': '#4d4d4d', 'PPP-AR/SDBS': '#1f77b4', 'PPP-AR/DCM': '#d62728', 'PPP-AR/IPC': '#2ca02c'}


def reference():
    values = {}
    for line in open(ROOT/'Data'/'sinex'/'IGS0OPSSNX_20233270000_01D_01D_CRD.SNX'):
        if line[7:11] in ('STAX', 'STAY', 'STAZ') and line[14:18] == 'MADR':
            values[line[7:11]] = float(line[47:68])
    return np.array([values['STAX'], values['STAY'], values['STAZ']])


def enu(frame, ref):
    x, y, z = ref
    lat, lon = np.arctan2(z, np.hypot(x, y)*(1-6.69437999014e-3)), np.arctan2(y, x)
    r = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                  [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return (frame[['x', 'y', 'z']].to_numpy(float)-ref)@r.T


def tag(name):
    return name.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')


def summary(metrics, group):
    s = metrics[metrics.session >= 0]
    day = metrics[metrics.session < 0].set_index(group)
    rows = []
    for key, g in s.groupby(group, sort=False):
        key = key if not isinstance(key, tuple) or len(group) > 1 else key[0]
        conv = g.convergence_min
        rows.append({**dict(zip(group, key if isinstance(key, tuple) else (key,))),
                     'sessions': len(g),
                     'converged': f'{conv.notna().sum()}/{len(g)}',
                     'conv_median': conv.median(), 'conv_mean': conv.mean(),
                     'conv_p68': conv.quantile(0.68),
                     'ttff_median': g.ttff_min.median(),
                     'fix_rate': 100*g.fix_rate.mean(), 'wrong_fix': 100*g.wrong_fix_rate.mean(),
                     'rms_e_cm': 100*np.sqrt(np.mean(g.rms_e**2)), 'rms_n_cm': 100*np.sqrt(np.mean(g.rms_n**2)),
                     'rms_u_cm': 100*np.sqrt(np.mean(g.rms_u**2)),
                     'day_fix': 100*day.loc[key, 'fix_rate'] if key in day.index else np.nan,
                     'day_final_3d_cm': 100*day.loc[key, 'final_3d'] if key in day.index else np.nan})
    return pd.DataFrame(rows)


def paired_tests(metrics, baseline, others, column='convergence_min', label='name'):
    """Wilcoxon signed-rank test of session-wise differences (unconverged sessions = 120 min)."""
    s = metrics[metrics.session >= 0].copy()
    s[column] = s[column].fillna(120.0)
    base = s[s[label] == baseline].set_index('session')[column]
    rows = []
    for other in others:
        o = s[s[label] == other].set_index('session')[column]
        common = base.index.intersection(o.index)
        d = (o.loc[common]-base.loc[common]).to_numpy()
        p = stats.wilcoxon(d, zero_method='zsplit').pvalue if len(d) > 5 and np.any(d != 0) else np.nan
        lo, hi = bootstrap_ci(d)
        rows.append(dict(comparison=f'{other} − {baseline}', n=len(d), median_difference=float(np.median(d)),
                         ci95_low=lo, ci95_high=hi, wilcoxon_p=p))
    return pd.DataFrame(rows)


def bootstrap_ci(d, n=5000, seed=1):
    if len(d) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(d, (n, len(d))), axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def plot_timeseries(grid, names, title_names, path, ref):
    fig, axes = plt.subplots(len(names), 1, figsize=(7, 1.6*len(names)), sharex=True, sharey=True)
    for ax, name, title in zip(np.atleast_1d(axes), names, title_names):
        f = OUT/grid/'positions'/f'{tag(name)}_s-1.csv'
        if not f.is_file():
            continue
        p = pd.read_csv(f)
        e = enu(p, ref)*100
        t = p.epok/3600
        for k, (lab, c) in enumerate((('East', '#1f77b4'), ('North', '#2ca02c'), ('Up', '#d62728'))):
            ax.plot(t, e[:, k], lw=0.6, color=c, label=lab)
        if 'status' in p:
            fixed = (p.status != 'float').to_numpy()
            ax.fill_between(t, -20, 20, where=fixed, color='#ffd54f', alpha=0.25, lw=0, label='fixed')
        ax.set_ylim(-20, 20)
        ax.set_ylabel('cm')
        ax.text(0.01, 0.92, title, transform=ax.transAxes, va='top', fontsize=8, fontweight='bold')
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=7, loc='upper right')
    np.atleast_1d(axes)[-1].set_xlabel('Hour of day (GPS time), 23 Nov 2023')
    fig.savefig(path)
    plt.close(fig)


def plot_sessions(grid, names, labels, path, ref, column='horizontal'):
    """Median and 68th percentile of the horizontal/vertical error over the 12 sessions."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for index, (name, label) in enumerate(zip(names, labels)):
        curves = []
        for k in range(12):
            f = OUT/grid/'positions'/f'{tag(name)}_s{2*k}.csv'
            if not f.is_file():
                continue
            p = pd.read_csv(f)
            e = enu(p, ref)
            t = p.epok.to_numpy()-p.epok.min()
            grid_t = np.arange(0, 7200, 30)
            curves.append(np.c_[np.interp(grid_t, t, np.hypot(e[:, 0], e[:, 1])), np.interp(grid_t, t, np.abs(e[:, 2]))])
        if not curves:
            continue
        c = np.array(curves)*100
        style = dict(color=COLORS.get(label.split(' ')[-1], f'C{index}'), lw=1.2)
        for j, ax in enumerate(axes):
            ax.plot(grid_t/60, np.median(c[:, :, j], axis=0), label=label, **style)
            ax.fill_between(grid_t/60, np.percentile(c[:, :, j], 25, axis=0), np.percentile(c[:, :, j], 75, axis=0),
                            alpha=0.12, color=style['color'], lw=0)
    for ax, title in zip(axes, ('Horizontal error', 'Vertical error')):
        ax.set_yscale('log'); ax.set_ylim(0.5, 200); ax.axhline(10, color='k', lw=0.6, ls='--')
        ax.set_xlabel('Time since start of session (min)'); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('cm (median, 25–75% band)')
    axes[1].legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def nl_products():
    """Histograms and table of the narrow-lane product test (Section 2.8) for each product set."""
    runs = {}
    for f in sorted((OUT/'nl_test').glob('*/nl_diagnostic/summary.json')):
        r = json.loads(f.read_text())
        name = Path(r['ar_products']).parent.name
        runs['GRG OSB + GFZ orbit/clock' if name == 'mixed_test' else name] = r['variants']
    order = [k for k in ('COD', 'GRG', 'WUM', 'GFZ', 'GRG OSB + GFZ orbit/clock') if k in runs]
    rows = [dict(products=k, variant=v, samples=runs[k][v]['samples'], arcs=runs[k][v]['arcs'],
                 median_abs=runs[k][v]['median_abs_fraction'], within_010=100*runs[k][v]['within_010'],
                 within_015=100*runs[k][v]['within_015']) for k in order for v in runs[k]]
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'nl_test'/'summary.csv', index=False)
    fig, axes = plt.subplots(1, len(order), figsize=(7.2, 1.9), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), order):
        for v, style in (('all models', dict(color='#1f77b4', alpha=0.7)), ('without new models', dict(color='#d62728', alpha=0.35))):
            h = np.asarray(runs[k][v]['histogram'], float)
            edges = np.linspace(-0.5, 0.5, len(h)+1)
            ax.bar(edges[:-1], 100*h/h.sum(), width=np.diff(edges), align='edge', label=v, **style)
        ax.axhline(100/len(h), color='k', lw=0.6, ls='--')
        ax.set_title(k.replace(' + ', '\n+ '), fontsize=7.5)
        ax.set_xlabel('NL fraction (cycle)', fontsize=7); ax.set_xticks([-0.4, 0, 0.4])
    np.atleast_1d(axes)[0].set_ylabel('%')
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.22))
    fig.savefig(FIG/'nl_test.png'); plt.close(fig)
    return table, pd.DataFrame()


def main(grid):
    if grid == 'nl':
        table, tests = nl_products()
        print(table.round(3).to_string(index=False))
        return
    ref = reference()
    metrics = pd.read_csv(OUT/grid/'metrics.csv')
    metrics['rms_h_cm'] = 100*np.hypot(metrics.rms_e, metrics.rms_n)
    if 'error' in metrics:
        print('errors:', metrics[metrics.error.notna()][['name', 'session', 'error']].to_string())
        metrics = metrics[metrics.error.isna()]
    result = {}
    if grid == 'products':
        table = summary(metrics, ['centre', 'technique'])
        table.to_csv(OUT/grid/'summary.csv', index=False)
        tests = pd.concat([paired_tests(metrics, f'{c} PPP', [f'{c} {t}' for t in ('PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC')], m)
                           .assign(metric=m) for m in ('convergence_min', 'rms_h_cm') for c in metrics.centre.unique()])
        tests.to_csv(OUT/grid/'tests.csv', index=False)
        best = 'COD'
        plot_timeseries(grid, [f'{best} {t}' for t in ('PPP', 'PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC')],
                        ['Float PPP', 'PPP-AR SDBS', 'PPP-AR DCM', 'PPP-AR IPC'], FIG/'products_timeseries_cod.png', ref)
        plot_sessions(grid, [f'{best} {t}' for t in ('PPP', 'PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC')],
                      ['PPP', 'PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC'], FIG/'products_sessions_cod.png', ref)
        # Convergence time per centre and technique
        s = metrics[metrics.session >= 0].copy(); s['conv'] = s.convergence_min.fillna(120)
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        centres = list(dict.fromkeys(s.centre)); techniques = ['PPP', 'PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC']
        for i, c in enumerate(centres):
            for j, t in enumerate(techniques):
                v = s[(s.centre == c) & (s.technique == t)].conv
                b = ax.boxplot(v, positions=[i*5+j], widths=0.7, patch_artist=True, showfliers=True)
                b['boxes'][0].set_facecolor(COLORS[t]); b['boxes'][0].set_alpha(0.6)
        ax.set_xticks([i*5+1.5 for i in range(len(centres))]); ax.set_xticklabels(centres)
        ax.set_ylabel('Convergence time (min)'); ax.set_ylim(0, 125)
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc=COLORS[t], alpha=0.6) for t in techniques], techniques, fontsize=7, ncol=4, loc='upper center')
        fig.savefig(FIG/'products_convergence_box.png'); plt.close(fig)
    elif grid == 'ablation':
        table = summary(metrics, ['name'])
        table.to_csv(OUT/grid/'summary.csv', index=False)
        names = list(dict.fromkeys(metrics.name))
        base = [n for n in names if n.startswith('S0')][0]
        tests = pd.concat([paired_tests(metrics, base, [n for n in names if n != base and n.startswith('S')], c)
                           .assign(metric=c) for c in ('convergence_min', 'ttff_min', 'fix_rate', 'wrong_fix_rate', 'rms_h_cm')])
        tests.to_csv(OUT/grid/'tests.csv', index=False)
        # Each component against the step it was added to (S5 builds on S3, not on S4)
        step = {n.split(' ')[0]: n for n in names}
        pairs = [('S1', 'S0'), ('S2', 'S1'), ('S2g', 'S2'), ('S3', 'S2'), ('S4', 'S3'), ('S5', 'S3'), ('S6', 'S5')]
        increments = pd.concat([paired_tests(metrics, step[b], [step[a]], c).assign(metric=c, step=a, base=b)
                                for a, b in pairs for c in ('convergence_min', 'ttff_min', 'fix_rate', 'wrong_fix_rate', 'rms_h_cm')])
        increments.to_csv(OUT/grid/'tests_increments.csv', index=False)
        print(increments[['step', 'base', 'metric', 'median_difference', 'ci95_low', 'ci95_high', 'wilcoxon_p']].round(4).to_string(index=False))
        steps = [n for n in names if n.startswith('S')]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
        s = metrics[metrics.session >= 0]
        for ax, (column, label, scale) in zip(axes, (('ttff_min', 'TTFF (min)', 1), ('fix_rate', 'Fix rate (%)', 100),
                                                     ('wrong_fix_rate', 'Wrong fixes (%)', 100))):
            data = [s[s.name == n][column].dropna().to_numpy()*scale for n in steps]
            ax.boxplot(data, widths=0.6)
            ax.set_xticks(range(1, len(steps)+1)); ax.set_xticklabels([n.split(' ')[0] for n in steps], fontsize=7, rotation=90)
            ax.set_title(label, fontsize=9)
        fig.savefig(FIG/'ablation_box.png'); plt.close(fig)
        plot_sessions(grid, ['PPP float (v2.9 models)', step['S0'], step['S2'], step['S4'], step['S5']],
                      ['PPP', 'S0 v2.8', 'S2 decision chain', 'S4 ML acceptance', 'S5 fixability'], FIG/'ablation_sessions.png', ref)
    else:
        table = summary(metrics, ['name'])
        table.to_csv(OUT/grid/'summary.csv', index=False)
        names = list(dict.fromkeys(metrics.name))
        tests = pd.concat([paired_tests(metrics, 'PPP float', [n for n in names if n != 'PPP float'], m).assign(metric=m)
                           for m in ('convergence_min', 'rms_h_cm')])
        tests.to_csv(OUT/grid/'tests.csv', index=False)
        best = [n for n in names if 'SDBS' in n]
        plot_sessions(grid, ['PPP float']+best, ['PPP']+[n.split(' ')[0] for n in best], FIG/'network_sessions.png', ref)
    pd.set_option('display.width', 250)
    print(table.round(2).to_string(index=False))
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
