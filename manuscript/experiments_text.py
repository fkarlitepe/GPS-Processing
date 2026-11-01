# Experiments and Conclusions. Tables are read from results/experiments/*/summary.csv and tests.csv.
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results'/'experiments'
FIG = RES/'figures'


def f(x, d=1):
    return '–' if pd.isna(x) else f'{x:.{d}f}'


def p(x):
    return '–' if pd.isna(x) else ('<0.001' if x < 0.001 else f'{x:.3f}')


def performance_rows(table, label):
    return [[label(r), r.converged, f(r.conv_median), f(r.ttff_median), f(r.fix_rate), f(r.wrong_fix),
             f(r.rms_e_cm, 2), f(r.rms_n_cm, 2), f(r.rms_u_cm, 2), f(r.day_fix)] for r in table.itertuples()]


PERF_HEADER = ['Solution', 'Conv.', 'Conv. (min)', 'TTFF (min)', 'Fix (%)', 'Wrong (%)', 'E (cm)', 'N (cm)', 'U (cm)', 'Fix 24 h (%)']


def short(comparison):
    """'COD PPP-AR/SDBS − COD PPP' -> 'COD: SDBS − PPP'; 'FCB-ekf PPP-AR/IPC − PPP float' -> 'EKF FCB: IPC − PPP'."""
    a, b = comparison.split(' − ')
    head, technique = a.rsplit(' ', 1)
    head = {'FCB-batch': 'Batch FCB', 'FCB-ekf': 'EKF FCB', 'FCB-ekf_ml': 'EKF+ML FCB'}.get(head, head)
    return f'{head}: {technique.replace("PPP-AR/", "")} − PPP'


def test_rows(tests, metric, measure):
    t = tests[tests.metric == metric]
    return [[short(r.comparison), measure, f(r.median_difference, 2), f'[{f(r.ci95_low, 2)}, {f(r.ci95_high, 2)}]', p(r.wilcoxon_p)]
            for r in t.itertuples()]


products = pd.read_csv(RES/'products'/'summary.csv')
products_tests = pd.read_csv(RES/'products'/'tests.csv')
network = pd.read_csv(RES/'network'/'summary.csv')
network_tests = pd.read_csv(RES/'network'/'tests.csv')
ablation = pd.read_csv(RES/'ablation'/'summary.csv')
ablation_tests = pd.read_csv(RES/'ablation'/'tests_increments.csv')
nl = pd.read_csv(RES/'nl_test'/'summary.csv')
validation = pd.concat([pd.read_csv(RES/'network_fcb'/'fcb'/f'validation_{m}.csv') for m in ('batch', 'ekf', 'ekf_ml')])


def increment_rows():
    names = {'S1': 'S1 physical models', 'S2': 'S2 decision chain', 'S2g': 'S2g PRIDE gates', 'S3': 'S3 class variance factors',
             'S4': 'S4 ML acceptance', 'S5': 'S5 fixability model', 'S6': 'S6 fix-and-hold'}
    scale = {'ttff_min': 1, 'fix_rate': 100, 'wrong_fix_rate': 100, 'rms_h_cm': 1}
    rows = []
    for (step, base), g in ablation_tests.groupby(['step', 'base'], sort=False):
        cells = [f'{names[step]} − {base}']
        for metric in scale:
            r = g[g.metric == metric].iloc[0]
            cells.append(f'{f(scale[metric]*r.median_difference, 1 if metric != "rms_h_cm" else 2)} ({p(r.wilcoxon_p)})')
        rows.append(cells)
    return rows


def nl_rows():
    rows = []
    for name, g in nl.groupby('products', sort=False):
        a = g[g.variant == 'all models'].iloc[0]
        b = g[g.variant == 'without new models'].iloc[0]
        rows.append([name, str(int(a.arcs)), f(100*a.median_abs/100, 3), f(a.within_010), f(a.within_015), f(b.within_010), f(b.within_015)])
    return rows


def validation_rows():
    rows = []
    for (method, system), g in validation.groupby(['method', 'system'], sort=False):
        n = g.ambiguities.sum()
        rms = (g.rms_cycles**2*g.ambiguities).sum()/n
        rows.append([{'batch': 'Batch', 'ekf': 'EKF', 'ekf+ml': 'EKF+ML', 'ekf_ml': 'EKF+ML'}.get(method, method),
                     {'G': 'GPS', 'E': 'Galileo'}[system], str(len(g)), str(int(n)), f(rms**0.5, 3),
                     f(100*(g.within_015*g.ambiguities).sum()/n), f(100*(g.within_025*g.ambiguities).sum()/n)])
    return rows


EXPERIMENTS = [
('h1', '3. Experiments'),

('h2', '3.1. Data, products and stations'),

('p', 'The test station is MADR (Madrid, Spain), an IGS station with a JAVAD TRE_G3TH receiver and an AOAD/M_T antenna. We used its 24 h RINEX 3 file of 23 November 2023 (day of year 327) with a 30 s interval. GPS was processed on L1/L2 (C1W/L1W and C2W/L2W) and Galileo on E1/E5a (C1X/L1X and C5X/L5X). The reference coordinate of MADR is the IGS daily combined SINEX solution of the same day, and every error in this section is the difference from this coordinate in east, north and up.'),

('p', 'Four MGEX product sets were used (Table {tab:products}). Each set contains the orbits, clocks, observable-specific biases (OSB), attitude quaternions and Earth rotation parameters of one analysis centre, so that the phase biases are always applied with the clocks they were estimated with. The satellite and receiver antenna corrections are from igs20_2287.atx, the ocean tide loading coefficients of MADR from the FES2004 BLQ table distributed with Net_Diff (Zhang, 2024), and the pole tide from the ERP file of the same centre; GRG does not provide an ERP file, so the CODE file was used. Net_Diff can download these products only through its Windows graphical interface, so we downloaded the same files from the Wuhan University IGS data centre, the archive used by Net_Diff, and stored them in the directory structure of our software (products per day and centre, models, SINEX and network observations in separate folders).'),

('tab:products', 'Product sets used in the experiments (23 November 2023). G and E: number of GPS and Galileo satellites with phase biases in the OSB file.',
 ['Set', 'Product line', 'Orbit / clock', 'Phase biases (G / E)', 'Galileo phase signals', 'Note'],
 [['COD', 'COD0MGXFIN', '5 min / 30 s', '33 / 25', 'L1C, L1X, L5Q, L5X', 'final'],
  ['GRG', 'GRG0MGXFIN', '5 min / 30 s', '32 / 23', 'L1C, L1X, L5Q, L5X', 'final, integer clocks; ERP of CODE'],
  ['WUM', 'WUM0MGXFIN', '5 min / 30 s', '31 / 23', 'L1C, L1X, L5Q, L5X', 'final'],
  ['GFZ', 'GFZ0MGXRAP', '5 min / 30 s', '31 / 23', 'L1C, L5C', 'rapid; no bias for the X signals of MADR']]),

('p', 'For the network experiment, the 24 h RINEX files of 50 IGS stations within about 2500 km of MADR were downloaded from the same archive (Figure {fig:map}). MADR itself is not part of the network. Two of the files (GOP6 and SFER) are RINEX 4 files, which the software does not read, so 48 stations were processed (46 with Galileo). All products and settings were the same as for the user station, with the COD product set providing orbits and clocks. The map was drawn with PyGMT v0.17 (Tian et al., 2025), the Python interface of GMT 6 (Wessel et al., 2019).'),

('fig:map', str(FIG/'station_map.png'), 'The user station MADR (star) and the IGS stations of the FCB network. Grey: RINEX 4 files that were not processed. Dashed circles: 500, 1000 and 2000 km from MADR.', 15.5),

('h2', '3.2. Experimental design and performance measures'),

('p', 'All solutions are static, with a 10° elevation mask for the float filter and 15° for ambiguity resolution, a zenith code and phase standard deviation of 0.3 m and 3 mm, and the error models of Section 2.2. To obtain repeated samples from one day, the 24 h file was divided into twelve independent 2 h sessions, and the filter was restarted at the beginning of each session. Every configuration was also run once over the whole day. For each session we computed:'),

('p', '• the **convergence time**, the first epoch after which the horizontal and the vertical error both stay below 10 cm for 20 consecutive epochs (10 min);'),
('p', '• the **time to first fix (TTFF)**, the first epoch after which the solution stays fixed for 20 consecutive epochs;'),
('p', '• the **fix rate**, the share of epochs with a fixed solution, and the **wrong-fix rate**, the share of fixed epochs after the first 30 min whose 3D error exceeds 10 cm;'),
('p', '• the **RMS of the east, north and up errors** in the last 30 min of the session.'),

('p', 'Sessions that did not converge were given a convergence time of 120 min. Two configurations were compared with the session-wise differences $d_k = x_k^{B} - x_k^{A}$, $k = 1, dotsaxis , 12$. Because twelve differences are too few to assume normality, we used the Wilcoxon signed-rank test (Wilcoxon, 1945) for the null hypothesis that the median difference is zero, and a percentile bootstrap with 5000 resamples (Efron and Tibshirani, 1993) for the 95% confidence interval of the median difference. A difference is called significant when $p < 0.05$. The sessions of one day are not fully independent, since they share the satellite geometry of the day and the same products, so the tests show whether a difference is consistent across sessions rather than how it would generalise to other days.'),

('h2', '3.3. Narrow-lane product test'),

('p', 'Before ambiguity resolution, each product set was checked with the test of Section 2.8 (Table {tab:nl}, Figure {fig:nl}). With the COD, GRG and WUM sets, 96–98% of the narrow-lane fractions lie within ±0.15 cycle and the median absolute fraction is about 0.03 cycle, close to the 91% within ±0.10 cycle reported by Ge et al. (2008). The GFZ set gives only 50% within ±0.15 cycle and many fractions near ±0.5 cycle. GFZ provides Galileo phase biases only for the C signals, which MADR does not track, so only GPS was tested; the GPS fractions of this rapid product are not consistent with its own clocks. As a negative control, the GRG phase biases were combined with the GFZ orbits and clocks. The software rejects this combination because the centres in the file headers differ; with the check bypassed, 64% of the fractions lie within ±0.15 cycle, and six GPS satellites have mean fractions close to ±0.5 cycle. The test therefore detects both an inconsistent product and a mixed product set.'),

('p', 'The models added in this work move fractions towards zero for all consistent sets: within ±0.10 cycle, from 84% to 91% (COD), from 85% to 94% (GRG) and from 82% to 85% (WUM). During this work, the test also revealed an error in our own software. The Earth rotation during the signal travel time was computed from $P/c$, which contains the receiver clock offset. At MADR this offset is −0.18 ms, which rotates the satellites by about 0.35 m and moved the 24 h float solution 7.5 cm to the east. With this error the fractions of even the COD set were uniformly distributed. After the correction (Section 2.2), the 24 h float solution agrees with the IGS coordinate within 0.4 cm in each component.'),

('tab:nl', 'Narrow-lane product test at MADR: number of arcs, median absolute fraction, and share of fractions within ±0.10 and ±0.15 cycle with all models and without the models added in this work (attitude, satellite PCV, ocean and pole tide, Shapiro delay).',
 ['Product set', 'Arcs', 'Median |frac.| (cycle)', '±0.10 (%)', '±0.15 (%)', '±0.10 old (%)', '±0.15 old (%)'], nl_rows()),

('fig:nl', str(FIG/'nl_test.png'), 'Distribution of the narrow-lane fractions at MADR for each product set, with all models (blue) and without the new models (red). Dashed line: uniform distribution.', 16),

('h2', '3.4. Experiment A: PPP and PPP-AR with analysis-centre products'),

('p', 'Table {tab:prodperf} summarises the twelve sessions for each product set and technique. With the three consistent sets, all sessions of all techniques converged. Float PPP needed a median of 19.5–29.5 min to reach 10 cm, and its horizontal error was still 1.5–3 cm after two hours, mostly in the east component (Figure {fig:prodsessions}). SDBS fixed the ambiguities after a median of 15 min in every set, fixed 84% of the epochs and reduced the median convergence time to 15 min. The RMS of the east error in the last 30 min dropped from 3.0 to 0.4–0.5 cm, and the north error from 1.5 to 0.4–0.6 cm. DCM and IPC also fixed, but later (TTFF 22–37 min; Figure {fig:prodbox}) and less often (58–64% of the epochs), and their accuracy lay between float PPP and SDBS. The vertical error changed by less than 0.4 cm. The up component is correlated with the tropospheric delay and the receiver clock, which are not fixed, so ambiguity resolution helps it less than the horizontal components.'),

('tab:prodperf', 'Experiment A: performance over twelve 2 h sessions (medians for times, means for rates, RMS over sessions for the last 30 min). Conv.: converged sessions. Fix 24 h: fix rate of the 24 h solution.',
 PERF_HEADER, performance_rows(products, lambda r: f'{r.centre} {r.technique.replace("PPP-AR/", "")}')),

('fig:prodsessions', str(FIG/'products_sessions_cod.png'), 'Experiment A, COD products: median horizontal and vertical error of the twelve sessions (lines) and their 25–75% range (bands). Dashed line: 10 cm.', 15.5),

('p', 'The statistical tests (Table {tab:prodtests}) show that the improvement in accuracy is consistent: for COD, GRG and WUM, all three PPP-AR techniques reduced the horizontal RMS in every comparison ($p <= 0.012$), by a median of 1.9–2.2 cm with SDBS and 0.8–1.9 cm with DCM and IPC. The shorter convergence time, by contrast, is not significant ($p >= 0.07$). In several sessions the float solution already reaches 10 cm within 15 min, and the first fixes appear at about the same time, so ambiguity resolution mainly improves the accuracy after convergence, not the time to reach 10 cm.'),

('tab:prodtests', 'Experiment A: Wilcoxon signed-rank tests of the session-wise differences from float PPP (negative: PPP-AR better). CI: 95% bootstrap interval of the median difference.',
 ['Comparison', 'Measure', 'Median diff.', '95% CI', 'p'],
 test_rows(products_tests, 'rms_h_cm', 'Horizontal RMS (cm)') + test_rows(products_tests, 'convergence_min', 'Convergence (min)')),

('p', 'The GFZ set shows what happens when the products are not consistent. The float solution is as good as with the other sets, but every PPP-AR technique fixed wrong integers in most sessions: 63–71% of the fixed epochs had a 3D error larger than 10 cm, and only 5–9 of the 12 sessions converged. The acceptance tests cannot detect this case, because the float ambiguities are biased by the product while their formal precision is high; the narrow-lane test of Section 3.3 detects it before any integer is fixed. In practice, an operator would stop at the test result of 50% within ±0.15 cycle, and the GFZ AR solutions are shown here only to illustrate this risk.'),

('fig:prodbox', str(FIG/'products_convergence_box.png'), 'Experiment A: convergence time of the twelve sessions for each product set and technique.', 15.5),

('h2', '3.5. Experiment B: PPP-AR with FCBs estimated from the network'),

('p', 'In the second experiment, no external bias products were used. The FCBs were estimated from the 48 network stations with the three estimators of Section 2.7, every 15 min, and applied at MADR with the COD orbits and clocks. Table {tab:loo} gives the leave-one-out validation. The batch estimator gives the smallest residuals at the left-out stations (RMS 0.057 cycle, 97% within ±0.15 cycle), followed by EKF+ML (0.075 cycle) and EKF (0.079 cycle). These values are similar to the 0.07 cycle reported for network products by Dai et al. (2019).'),

('tab:loo', 'Leave-one-out validation of the network FCBs: number of left-out stations and ambiguities, RMS of the remaining fractions, and share within ±0.15 and ±0.25 cycle.',
 ['Estimator', 'System', 'Stations', 'Ambiguities', 'RMS (cycle)', '±0.15 (%)', '±0.25 (%)'], validation_rows()),

('p', 'The network FCBs are available only from 03:15. A narrow-lane value is used only when its float standard deviation is below 0.2 cycle, and with a network that starts at 00:00 this takes about three hours. The first session (00:00–02:00) therefore has no FCB and was excluded, and the second session could fix only after 03:15. Analysis centres avoid this gap by processing several days in one solution. Table {tab:netperf} and Figure {fig:netsessions} show the results of the remaining eleven sessions. With all three estimators, SDBS fixed after a median of 15 min, as with the COD OSB products, but less often (73–78% of the epochs) and with more wrong fixes (3–7%). The horizontal RMS improved by a median of 0.3–0.8 cm compared with float PPP. The improvement is significant for the EKF and EKF+ML FCBs with SDBS ($p = 0.024$) and with IPC for EKF ($p = 0.042$), but not for the batch FCBs ($p >= 0.05$; Table {tab:nettests}). With the batch FCBs, IPC also converged later than float PPP (median +15 min, $p = 0.032$).'),

('tab:netperf', 'Experiment B: performance with network FCBs over the eleven sessions that have FCBs (same measures as Table {tab:prodperf}).',
 PERF_HEADER, performance_rows(network, lambda r: short(r.name + ' − x').split(' − ')[0] if r.name != 'PPP float' else 'Float PPP')),

('fig:netsessions', str(FIG/'network_sessions.png'), 'Experiment B: median horizontal and vertical error of the sessions for float PPP and SDBS with the FCBs of the three network estimators.', 15.5),

('p', 'The ranking at the user station is the opposite of the leave-one-out ranking. The batch FCBs fit the network best, but at MADR they gave the most wrong fixes (7.0%) and the smallest gain in accuracy, while the EKF FCBs gave 3.2% wrong fixes and the EKF and EKF+ML FCBs fixed 78% of the epochs. A likely reason is that the batch solution estimates each 15 min block independently, so a block with few contributing stations can produce an FCB that fits these stations but jumps relative to its neighbours; the EKF carries the previous blocks forward and smooths such jumps. In addition, the left-out station still shares the satellites and the time blocks of the network, so its residuals reward a close fit to the network. The user station, which is not part of the network, is a stricter test. Compared with the COD OSB products (Experiment A), the network FCBs give a lower fix rate (78% versus 84%) and smaller gains in accuracy, because the products of an analysis centre come from a global network processed over several days.'),

('tab:nettests', 'Experiment B: Wilcoxon signed-rank tests of the session-wise differences from float PPP.',
 ['Comparison', 'Measure', 'Median diff.', '95% CI', 'p'],
 test_rows(network_tests, 'rms_h_cm', 'Horizontal RMS (cm)') + test_rows(network_tests, 'convergence_min', 'Convergence (min)')),

('h2', '3.6. Contribution of each new component'),

('p', 'To measure what each change adds, the COD/SDBS solution was rebuilt step by step, starting from the previous version of the software (S0: ratio test with a fixed threshold of 3, at least two ambiguities, no satellite exclusion, no attitude, satellite PCV, ocean and pole tide or Shapiro delay). Each step adds one component (Table {tab:ablation}): S1 the physical models of Section 2.2, S2 the decision chain of Section 2.5 (Dong–Bock ordering, partial fixing with the bootstrapped success rate, FFRT and satellite exclusion), S3 the variance factor per observation group, S4 the acceptance test trained with biases, S5 the fixability model (on S3) and S6 fix-and-hold (on S5). S2g adds the PRIDE candidate gates to S2. Table {tab:increments} tests each step against the step it was added to.'),

('tab:ablation', 'Contribution of each component (COD products, SDBS, twelve 2 h sessions; same measures as Table {tab:prodperf}).',
 PERF_HEADER, performance_rows(ablation, lambda r: r.name.replace(' (P0, PAR, FFRT, exclusion)', '').replace(' (ratio test, basic models)', '').replace(' (v2.9 models)', ''))),

('tab:increments', 'Change caused by each component relative to the step it was added to: median of the session-wise differences and, in brackets, the Wilcoxon p value. Sessions without a fix enter the TTFF with 120 min.',
 ['Step − base', 'TTFF (min)', 'Fix rate (%)', 'Wrong fixes (%)', 'Horizontal RMS (cm)'], increment_rows()),

('fig:ablation', str(FIG/'ablation_box.png'), 'Time to first fix, fix rate and wrong-fix rate of the twelve sessions for each step of Table {tab:ablation}.', 15.5),

('p', '**Decision chain.** The largest change comes from S2. With the ratio test (S0 and S1), the integers were accepted in only 1% of the epochs, and when they were, they were often wrong: in one S0 session 96% of the fixed epochs were wrong. With the decision chain the fix rate rose to 84%, the TTFF dropped to 15 min in every session, and the horizontal RMS fell by a median of 1.7 cm ($p < 0.001$). A fixed ratio threshold rejects correct solutions when many ambiguities are fixed together, because the ratio of the two best candidates is then close to one even if the best candidate is correct; partial fixing with the success rate and an FFRT threshold computed for the current ambiguity covariance avoids this (Verhagen et al., 2011, 2013).'),

('p', '**Physical models.** S1 changed the fix rate very little, because the ratio test remained the bottleneck, but it reduced the error of the float solution: the up RMS from 2.8 to 1.9 cm and the 3D error of the 24 h solution from 0.44 to 0.24 cm. The horizontal RMS decreased by a median of 0.5 cm, which is not significant with twelve sessions ($p = 0.064$). The same models raised the share of narrow-lane fractions within ±0.10 cycle by 4–9 percentage points (Section 3.3), so their main role is to make the float ambiguities closer to integers.'),

('p', '**PRIDE candidate gates.** Adding the fraction and standard deviation gates of PRIDE PPP-AR to S2 (S2g) delayed the first fix from 15 to 53 min ($p < 0.001$), lowered the fix rate from 84% to 33% and increased the horizontal RMS by 0.8 cm. These gates were designed for a batch solution, in which every ambiguity has already converged. In a filter, the fractions of a converging arc are still large, so the gates remove exactly the arcs that the success rate and the FFRT would have fixed correctly. For this reason they are off by default.'),

('p', '**Variance factors per observation group.** S3 did not change the convergence time or the TTFF. It lowered the fix rate by 0.6 percentage points ($p = 0.008$) and the mean wrong-fix rate from 2.2% to 1.4%, with the largest wrong-fix rate of a session falling from 17% to 7%; the latter change is not significant, because only three sessions had wrong fixes at all. MADR is a static station with one receiver and a well-known noise level, so the fixed weights of S2 were already close to correct. The variance factors are expected to matter more for kinematic data and receivers of different quality, which were not tested here.'),

('p', '**Acceptance test trained with biases.** S4 gave the lowest mean wrong-fix rate of all steps (0.7%), at the cost of a later first fix (median +2.5 min, $p = 0.016$) and a lower fix rate (−5 percentage points, $p = 0.004$). The horizontal accuracy did not change. The model is thus more conservative than the FFRT alone: it rejects some correct fixes in order to reject the biased ones. Whether this trade is worthwhile depends on the application; for monitoring, where a wrong fix is worse than a late one, S4 is the preferred setting.'),

('p', '**Fixability model and fix-and-hold.** S5 made the solution worse: the TTFF increased by a median of 20 min, the fix rate dropped by 14 percentage points and the convergence time increased by 10.5 min (all $p < 0.01$). The model is trained on arcs that converged later in the same session. In a 2 h session, few arcs reach the tight limit ($%sigma < 0.05$ cycle) needed for a label, so the model is probably trained on too few positive labels and rates many candidates below the threshold of 0.5. In the 24 h solution, where more labels are available, its fix rate (90%) was close to that of S2 (93%). The idea therefore needs either longer data or a model trained on other days before it can help short sessions. Fix-and-hold (S6) did not change the medians, but in two sessions a wrong fix was held and the solution did not converge, which raised the mean wrong-fix rate to 15%. Fix-and-hold is therefore only safe together with a strict acceptance test.'),

('fig:ablsessions', str(FIG/'ablation_sessions.png'), 'Median horizontal and vertical error of the twelve sessions for float PPP, the previous version (S0), the decision chain (S2), the ML acceptance test (S4) and the fixability model (S5).', 15.5),

('p', 'In summary, two of the new components improve the solution clearly and in every test: the physical models, which make the narrow-lane fractions and the float solution more accurate, and the decision chain, which turns a 1% fix rate into an 84% fix rate with a 15 min TTFF. The bias-aware acceptance test halved the mean wrong-fix rate at a small cost in availability, although wrong fixes were too rare in these sessions for this difference to be significant. The variance factors per group did not help at a static station, and the fixability model and fix-and-hold made short sessions worse. The product test of Section 2.8 is not part of the ablation, but Experiment A shows its value: it identifies the GFZ set, with which every PPP-AR technique produced mostly wrong fixes, before any integer is fixed.'),

('h1', '4. Conclusions'),

('p', 'We presented an undifferenced GPS+Galileo PPP/PPP-AR software in which float PPP and three PPP-AR user models (DCM, SDBS and IPC) share one Kalman filter, one set of error models and one integer decision, and tested it with 24 h of data from the IGS station MADR, four analysis-centre product sets and FCBs estimated from a network of 48 European IGS stations. The main results are as follows.'),

('p', '1. With consistent analysis-centre products (COD, GRG and WUM), PPP-AR reduced the horizontal RMS error in the last 30 min of 2 h sessions from 3.3 cm (float) to 0.6–0.8 cm with SDBS, and the improvement was significant for every product set and every technique ($p <= 0.012$). SDBS fixed after 15 min and fixed 84% of the epochs; DCM and IPC fixed later and less often. The time to reach 10 cm was shorter but not significantly so, because the float solution often reached this level at the same time as the first fix.'),

('p', '2. The three user models give the same position once the ambiguities are fixed (Shi and Gao, 2014), but in a filter they behave differently before that. In our implementation the between-satellite model fixed earliest and most often. A likely reason is that differencing removes the receiver clocks exactly, whereas DCM estimates three clocks per system as white noise and IPC fixes the wide lanes only from arc means.'),

('p', '3. Inconsistent products cannot be detected by the acceptance tests. With the GFZ rapid products, 63–71% of the fixed epochs were wrong. The narrow-lane test, which needs only a known coordinate and one day of data, separated the consistent sets (96–98% of the fractions within ±0.15 cycle) from the GFZ set (50%) and from a mixed set (64%), and the header check rejected the mixed set. The test also revealed an error in our own Earth-rotation correction, which had moved the float solution by 7.5 cm.'),

('p', '4. FCBs estimated from 48 stations with the EKF estimators improved the horizontal accuracy at MADR significantly ($p = 0.024$), but less than the analysis-centre products, and they were available only after three hours of network data. The estimator with the best leave-one-out statistics (batch) produced the most wrong fixes at the user station, so leave-one-out residuals alone are not sufficient to choose an estimator.'),

('p', '5. The step-by-step comparison shows that most of the gain over the previous version comes from the decision chain (Dong–Bock ordering, partial fixing by success rate, FFRT and satellite exclusion) and from the physical models. The bias-aware ML acceptance test halved the mean wrong-fix rate (from 1.4% to 0.7%, not significant) at a cost of 5 percentage points in fix rate. The PRIDE candidate gates, the fixability model in 2 h sessions and fix-and-hold made the solution worse and are therefore not enabled by default.'),

('p', 'These results come from one static station and one day, so they show how the components behave relative to each other, not how they would perform in general. Future work will test the variance factors and the fixability model on kinematic data and multi-day series, include more stations and days, and process the FCB network continuously over several days so that the biases are available from the start of the day.'),
]
