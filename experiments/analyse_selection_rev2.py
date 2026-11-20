"""Statistics of the second PPP selection revision (three days, GE and GEC).

    python3 experiments/analyse_selection_rev2.py  ->  results/experiments/selection_rev2/summary.md, tables.json

Unit of inference: the station. For a comparison A - B, the paired session differences are averaged per station
(8 stations, 18 sessions each: six 2-h sessions on each of three days); the p-value is the exact Wilcoxon signed-rank test of these 8 station means and the
95% interval a station bootstrap (5000 resamples) of the Hodges-Lehmann shift of the session differences.
Sessions that do not converge within 2 h are censored: Kaplan-Meier medians and a log-rank test are given for
convergence. Holm correction within each family (WDOP vs DOP: 12 tests per outcome; controls; GEC).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT/'results'/'experiments'/'selection_rev2'
rng = np.random.default_rng(11)
CENSOR = 120.0


def md(df, index=True):
    """Markdown table without the optional 'tabulate' package."""
    df = df.reset_index() if index else df
    cols = [' / '.join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    rows += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(rows)


def load(systems):
    f = T/f'sessions_{systems}.csv'
    if f.exists():
        d = pd.read_csv(f)
    else:
        parts = sorted((T/'parts').glob(f'{systems}_*.pkl'))
        d = pd.concat([pd.read_pickle(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
    if d.empty:
        return d
    if 'error' in d:
        d = d[d.error.isna()].copy()
    d['conv'] = d.convergence_min.fillna(CENSOR).clip(upper=CENSOR)
    d['event'] = d.convergence_min.notna()
    d['rms_mm'] = d.rms_3d*1e3
    return d


def hl(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i]+x[j])/2))


def km_median(t, event):
    order = np.argsort(t)
    t, event = np.asarray(t)[order], np.asarray(event)[order]
    s, n = 1.0, len(t)
    for i, (ti, ei) in enumerate(zip(t, event)):
        if ei:
            s *= 1-1/(n-i)
            if s <= 0.5:
                return float(ti)
    return np.nan


def logrank(t1, e1, t2, e2):
    t = np.r_[t1, t2]
    e = np.r_[e1, e2].astype(bool)
    g = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    o_minus_e, var = 0.0, 0.0
    for ti in np.unique(t[e]):
        at = t >= ti
        n, n1 = at.sum(), (at & (g == 0)).sum()
        d = ((t == ti) & e).sum()
        d1 = ((t == ti) & e & (g == 0)).sum()
        o_minus_e += d1-d*n1/n
        if n > 1:
            var += d*(n1/n)*(1-n1/n)*(n-d)/(n-1)
    return float(stats.chi2.sf(o_minus_e**2/var, 1)) if var > 0 else 1.0


def paired(d, a, b, col, key=('day', 'station', 'session')):
    key = list(key)
    m = d[d.setting_k == a][key+[col, 'event']].merge(d[d.setting_k == b][key+[col, 'event']], on=key, suffixes=('_a', '_b'))
    if m.empty:
        return None
    diff = (m[col+'_a']-m[col+'_b']).to_numpy()
    per = m.assign(diff=diff).groupby('station')['diff'].mean()
    p = stats.wilcoxon(per).pvalue if len(per) >= 5 and np.any(per != 0) else 1.0
    stations = m.station.unique()
    groups = {s: diff[(m.station == s).to_numpy()] for s in stations}
    boot = [hl(np.concatenate([groups[s] for s in rng.choice(stations, len(stations))])) for _ in range(2000)]
    out = dict(a=a, b=b, outcome=col, sessions=len(m), stations=len(per), shift=hl(diff), ci=np.percentile(boot, [2.5, 97.5]).tolist(),
               p=float(p), better=float(np.mean(per < 0)), median_a=float(np.median(m[col+'_a'])), median_b=float(np.median(m[col+'_b'])))
    if col == 'conv':
        out.update(km_a=km_median(m.conv_a, m.event_a), km_b=km_median(m.conv_b, m.event_b),
                   logrank_p=logrank(m.conv_a, m.event_a, m.conv_b, m.event_b))
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    adj, run = np.empty_like(ps), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, (len(ps)-rank)*ps[i])
        adj[i] = min(1.0, run)
    return adj


def family(d, pairs):
    rows = []
    for a, b in pairs:
        for col in ('rms_mm', 'conv'):
            r = paired(d, a, b, col)
            if r:
                rows.append(r)
    t = pd.DataFrame(rows)
    if not t.empty:
        for col in ('rms_mm', 'conv'):
            m = t.outcome == col
            t.loc[m, 'p_holm'] = holm(t.loc[m, 'p'].to_numpy())
    return t


def table(d):
    g = d.groupby('setting_k').agg(k=('k', 'first'), setting=('setting', 'first'), conv=('conv', 'median'),
                                   rms=('rms_mm', 'median'), converged=('event', 'mean'), pdop=('median_pdop', 'median'),
                                   wpdop=('median_wpdop', 'median'), changes=('reselections', 'mean'),
                                   changes_median=('reselections', 'median'), low=('low_selected', 'mean'),
                                   low_all=('low_all', 'mean'), visible=('mean_visible', 'mean'), used=('mean_used', 'mean'),
                                   evals=('evaluations', 'mean'), sel_ms=('selection_ms', 'mean'), n=('rms_mm', 'size')).reset_index()
    g['km'] = [km_median(d[d.setting_k == s].conv, d[d.setting_k == s].event) for s in g.setting_k]
    return g.sort_values(['k', 'setting'])


def main():
    out, lines = {}, ['# PPP satellite selection, revision 2\n']
    for systems in ('GE', 'GEC'):
        d = load(systems)
        if d.empty:
            continue
        d['setting_k'] = d.setting+'@'+d.k.astype(str)
        d.loc[d.k == 0, 'setting_k'] = d.setting
        tab = table(d)
        out[f'{systems}_table'] = tab.to_dict('records')
        lines.append(f'## {systems}: {d.station.nunique()} stations, {d.day.nunique()} days, {len(d)} session runs\n')
        lines.append(md(tab.round(2), index=False)+'\n')
        ks = sorted(k for k in d.k.unique() if k)
        if systems == 'GE':
            wd = [(f'FIBO-GNSS2|{w}@{k}', f'FIBO-GNSS2|none@{k}') for k in ks for w in ('elevation', 'cn0', 'filter')]
            wd += [(f'BACKWARD+LS|filter@{k}', f'BACKWARD+LS|none@{k}') for k in ks]
            ctl = []
            for k in ks:
                ctl += [(f'FIBO-GNSS2|filter@{k}', 'ALL'), (f'FIBO-GNSS2|none@{k}', f'BACKWARD+LS|none@{k}'),
                        (f'FIBO-GNSS2|filter@{k}', f'BACKWARD+LS|filter@{k}'),
                        (f'FIBO-GNSS2|none|hold600@{k}', f'FIBO-GNSS2|none@{k}'),
                        (f'FIBO-GNSS2|filter|hold600@{k}', f'FIBO-GNSS2|filter@{k}'),
                        (f'FIBO-GNSS2|filter|hold600@{k}', f'FIBO-GNSS2|none|hold600@{k}'),
                        (f'FIBO-GNSS2|filter|hold600@{k}', f'RANDOM|none|hold600@{k}'),
                        (f'FIBO-GNSS2|none@{k}', f'RANDOM|none|hold600@{k}'),
                        (f'FIBO-GNSS2|filter|reuse@{k}', f'FIBO-GNSS2|filter@{k}'),
                        (f'FIBO-GNSS2|filter|drop@{k}', f'FIBO-GNSS2|filter@{k}')]
            ctl += [(f'FIBO-GNSS2|filter|h={h}@8', 'FIBO-GNSS2|filter@8') for h in ('0.0', '0.1')]
            ctl += [(f'FIBO-GNSS2|filter|hold{t}@8', 'FIBO-GNSS2|filter|hold600@8') for t in (300, 1800)]
            ctl += [('FIBO-GNSS2|filter@8', 'FIBO-GNSS+LS|filter@8')]
            fams = {'wdop': wd, 'controls': ctl}
        else:
            ctl = []
            for k in ks:
                ctl += [(f'FIBO-GNSS2|filter@{k}', 'ALL'), (f'FIBO-GNSS2|filter@{k}', f'FIBO-GNSS2|none@{k}'),
                        (f'BACKWARD+LS|filter@{k}', f'BACKWARD+LS|none@{k}'),
                        (f'FIBO-GNSS2|filter|hold600@{k}', f'FIBO-GNSS2|filter@{k}'),
                        (f'FIBO-GNSS2|filter|hold600@{k}', f'RANDOM|none|hold600@{k}'),
                        (f'FIBO-GNSS2|filter|drop@{k}', f'FIBO-GNSS2|filter@{k}')]
            ctl += [('FIBO-GNSS2|filter|kinematic@12', 'ALL|kinematic'), ('FIBO-GNSS2|filter|kinematic@12', 'FIBO-GNSS2|none|kinematic@12'),
                    ('FIBO-GNSS2|filter|hold600|kinematic@12', 'FIBO-GNSS2|filter|kinematic@12')]
            fams = {'controls': ctl}
        for name, pairs in fams.items():
            t = family(d, pairs)
            out[f'{systems}_{name}'] = t.to_dict('records')
            if not t.empty:
                tt = t.copy()
                tt['ci'] = tt.ci.apply(lambda c: f'[{c[0]:.1f}, {c[1]:.1f}]')
                lines.append(f'### {systems} {name} (station-level Wilcoxon, station-bootstrap CI, Holm per outcome)\n')
                cols = ['a', 'b', 'outcome', 'shift', 'ci', 'p', 'p_holm', 'better', 'median_a', 'median_b']+(['km_a', 'km_b', 'logrank_p'] if 'km_a' in tt else [])
                lines.append(md(tt[cols].round(3), index=False)+'\n')
        # per-day consistency of the WDOP effect
        if systems == 'GE':
            rows = []
            for day, g in d.groupby('day'):
                for k in ks:
                    a = g[g.setting_k == f'FIBO-GNSS2|filter@{k}'].rms_mm.median()
                    b = g[g.setting_k == f'FIBO-GNSS2|none@{k}'].rms_mm.median()
                    c = g[g.setting_k == 'ALL'].rms_mm.median()
                    rows.append(dict(day=day, k=k, wdop=a, dop=b, all=c))
            out['GE_per_day'] = rows
            lines.append('### GE per day (median 3D RMS, mm)\n'+md(pd.DataFrame(rows).round(1), index=False)+'\n')
        # correlation of DOP with the PPP outcome
        sel = d[d.k > 0]
        out[f'{systems}_spearman'] = {c: float(stats.spearmanr(sel[c], sel.rms_mm, nan_policy='omit').statistic)
                                      for c in ('median_pdop', 'median_wpdop', 'mean_sigma_u')}
    tf = T/'timing.csv'
    if tf.exists():
        t = pd.read_csv(tf)
        t['filter_ms'] = t.core_ms+t.update_ms
        t['total_ms'] = t.filter_ms+t.selection_ms
        g = t.groupby(['systems', 'k', 'setting']).agg(visible=('visible', 'mean'), used=('used', 'mean'), states=('states', 'mean'),
                                                       selection=('selection_ms', 'mean'), core=('core_ms', 'mean'),
                                                       update=('update_ms', 'mean'), filter=('filter_ms', 'mean'),
                                                       total=('total_ms', 'mean')).reset_index()
        for systems in g.systems.unique():
            base = g[(g.systems == systems) & (g.setting == 'ALL')].total.iloc[0]
            g.loc[g.systems == systems, 'relative'] = g.loc[g.systems == systems, 'total']/base
        out['timing'] = g.to_dict('records')
        lines.append('## Run time per epoch (ms)\n'+md(g.round(2), index=False)+'\n')
    (T/'summary.md').write_text('\n'.join(lines))
    json.dump(out, open(T/'tables.json', 'w'), indent=1, default=float)
    print('\n'.join(lines)[:4000])


if __name__ == '__main__':
    main()
