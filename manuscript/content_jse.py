# ASCE Journal of Surveying Engineering version (Technical Paper). Build: python3 manuscript/build.py jse
# Tables are read from results/experiments/*/summary.csv, tests.csv and tests_increments.csv.
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results'/'experiments'
FIG = RES/'figures'


def f(x, d=1):
    return '–' if pd.isna(x) else f'{x:.{d}f}'


def p(x):
    return '–' if pd.isna(x) else ('<0.001' if x < 0.001 else f'{x:.3f}')


products = pd.read_csv(RES/'products'/'summary.csv')
products_tests = pd.read_csv(RES/'products'/'tests.csv')
network = pd.read_csv(RES/'network'/'summary.csv')
network_tests = pd.read_csv(RES/'network'/'tests.csv')
ablation = pd.read_csv(RES/'ablation'/'summary.csv')
increments = pd.read_csv(RES/'ablation'/'tests_increments.csv')
nl = pd.read_csv(RES/'nl_test'/'summary.csv')


def performance(r):
    return [r.converged, f(r.conv_median), f(r.ttff_median), f(r.fix_rate), f(r.wrong_fix),
            f(np.hypot(r.rms_e_cm, r.rms_n_cm), 2), f(r.rms_u_cm, 2)]


PERF = ['Conv.', 'Conv. time (min)', 'TTFF (min)', 'Fix (%)', 'Wrong fixes (%)', 'Hor. RMS (cm)', 'Up RMS (cm)']
FCB = {'FCB-batch': 'Batch', 'FCB-ekf': 'EKF', 'FCB-ekf_ml': 'EKF+ML'}


def label(name):
    """'COD PPP-AR/SDBS' -> 'COD SDBS'; 'FCB-ekf PPP-AR/IPC' -> 'EKF IPC'; 'PPP float' -> 'Float PPP'."""
    if name == 'PPP float':
        return 'Float PPP'
    head, technique = name.rsplit(' ', 1)
    return f'{FCB.get(head, head)} {technique.replace("PPP-AR/", "")}'


def product_rows():
    return [[label(f'{r.centre} {r.technique}')]+performance(r) for r in products.itertuples()]


def network_rows():
    return [[label(r.name)]+performance(r) for r in network.itertuples()]


def test_rows(tests):
    rows = []
    for metric, measure in (('rms_h_cm', 'Hor. RMS (cm)'), ('convergence_min', 'Conv. time (min)')):
        for r in tests[tests.metric == metric].itertuples():
            a, b = r.comparison.split(' − ')
            rows.append([f'{label(a)} − {label(b)}', measure, f(r.median_difference, 2),
                         f'[{f(r.ci95_low, 2)}, {f(r.ci95_high, 2)}]', p(r.wilcoxon_p)])
    return rows


STEPS = {'S0': 'S0 previous version', 'S1': 'S1 + physical models', 'S2': 'S2 + decision chain',
         'S2g': 'S2g = S2 + PRIDE gates', 'S3': 'S3 + group variance factors', 'S4': 'S4 + ML acceptance',
         'S5': 'S5 = S3 + fixability model', 'S6': 'S6 + fix-and-hold', 'PPP': 'Float PPP'}


def ablation_rows():
    return [[STEPS[r.name.split(' ')[0]]]+performance(r) for r in ablation.itertuples()]


def increment_rows():
    rows = []
    for (step, base), g in increments.groupby(['step', 'base'], sort=False):
        cells = [f'{step} − {base}']
        for metric, scale, d in (('ttff_min', 1, 1), ('fix_rate', 100, 1), ('wrong_fix_rate', 100, 1), ('rms_h_cm', 1, 2)):
            r = g[g.metric == metric].iloc[0]
            cells.append(f'{f(scale*r.median_difference, d)} ({p(r.wilcoxon_p)})')
        rows.append(cells)
    return rows


def loo_rows():
    rows = []
    v = pd.concat([pd.read_csv(RES/'network_fcb'/'fcb'/f'validation_{m}.csv') for m in ('batch', 'ekf', 'ekf_ml')])
    for (method, system), g in v.groupby(['method', 'system'], sort=False):
        n = g.ambiguities.sum()
        rows.append([{'batch': 'Batch', 'ekf': 'EKF', 'ekf+ml': 'EKF+ML'}[method], {'G': 'GPS', 'E': 'Galileo'}[system],
                     str(len(g)), str(int(n)), f(((g.rms_cycles**2*g.ambiguities).sum()/n)**0.5, 3),
                     f(100*(g.within_015*g.ambiguities).sum()/n), f(100*(g.within_025*g.ambiguities).sum()/n)])
    return rows


def nl_rows():
    rows = []
    for name, g in nl.groupby('products', sort=False):
        a = g[g.variant == 'all models'].iloc[0]
        b = g[g.variant == 'without new models'].iloc[0]
        rows.append([name.replace('GRG OSB + GFZ orbit/clock', 'GRG biases + GFZ orbit/clock'), str(int(a.arcs)),
                     f(a.median_abs, 3), f(a.within_010), f(a.within_015), f(b.within_010)])
    return rows


BLOCKS = [
('title', 'Supporting GPS+Galileo PPP Ambiguity Resolution with an Extended Kalman Filter and Machine Learning'),
('center', 'Furkan Karlıtepe¹'),
('center', '¹Dept. of Geomatics Engineering, Tokat Gaziosmanpaşa Univ., Tokat, Türkiye. Email: furkan.karlitepe@gop.edu.tr'),
('break',),

('h1', 'Abstract'),
('p', 'Precise point positioning with ambiguity resolution (PPP-AR) gives centimetre-level coordinates from a single receiver, but wrong integer fixes and inconsistent satellite products limit its reliability. This study presents a GPS+Galileo PPP-AR software in which float PPP and three user models (decoupled clock, between-satellite single difference and integer phase clock) share one extended Kalman filter (EKF) and one integer decision chain, supported by a machine-learning acceptance test and a narrow-lane product check. Twelve 2-h sessions of IGS station MADR were processed with the products of four analysis centres and with fractional cycle biases estimated from 48 European stations. With consistent products, PPP-AR reduced the horizontal RMS error from 3.2–3.5 to 0.6–0.8 cm and fixed 84% of the epochs with the single-difference model. An inconsistent product set gave 63–71% wrong fixes and was detected beforehand by the narrow-lane check. The decision chain gave most of the improvement, and the machine-learning test halved the wrong fixes, although not significantly. Consistent products and a strict acceptance rule are therefore the keys to reliable PPP-AR in surveying.'),
('p', '**Practical Applications:** Surveyors who use PPP for control points, monitoring or static positioning want centimetre-level coordinates from short occupations without a nearby base station. This study shows that, with consistent satellite products, a 2-h GPS+Galileo occupation processed with ambiguity resolution gives horizontal errors below 1 cm in most sessions, compared with about 3 cm for the standard float solution. The choice of products matters more than the choice of user model: products whose phase biases do not match the orbits and clocks produced wrong integers in most sessions, while the formal quality indicators of the solution looked normal. A simple test, which needs only a station with a known coordinate and one day of data, detects such products before they are used. The study also shows that the rule used to accept the integers has a larger effect on the result than the extra machine-learning components, and that some options that are common in batch software, such as strict fraction gates and fix-and-hold, delay or degrade short static sessions. The software and processing scripts are openly available online.'),
('p', '**Author keywords:** Precise point positioning; Ambiguity resolution; GNSS; Galileo; Fractional cycle bias; Kalman filter; Product consistency.'),
('break',),

('h1', 'Introduction'),
('p', 'Precise point positioning (PPP) gives centimetre-level coordinates from one receiver and precise satellite orbits and clocks, but the float solution needs tens of minutes to converge, and the undifferenced phase ambiguities are not integers because hardware delays are absorbed into them. PPP with ambiguity resolution (PPP-AR) removes these delays with satellite bias products and shortens the time to centimetre accuracy (Ge et al. 2008; Collins et al. 2010; Laurichesse and Blot 2016). Previous studies have shown its value for surveying, for example as a fill-in for sparse CORS networks (Grinter et al. 2020), in static surveys under forest canopy (Konakoglu and Yilmaz 2024) and with regional network augmentation (Wang et al. 2020), and Martín et al. (2011) showed that PPP results depend on the products and software used. Whether PPP-AR works in practice depends on three conditions: a sound integer estimator, an acceptance rule that keeps wrong fixes rare, and bias products that belong to the same orbit and clock solution as the one the user applies.'),
('p', 'The integer estimator is well established: integer least squares solved with LAMBDA gives the highest success rate (Teunissen 1995). The acceptance rule is less settled. Many PPP-AR implementations accept the integers with a fixed ratio threshold (Li et al. 2016) or with fixed fraction and precision gates (Geng et al. 2019). Verhagen et al. (2011) proposed choosing the subset to fix by its bootstrapped success rate and testing it with a fixed failure-rate ratio test (FFRT), and Ma et al. (2021) evaluated such partial fixing for PPP in flight tests. These criteria assume unbiased float ambiguities, whereas real float ambiguities are biased by multipath, unmodelled satellite attitude and product errors.'),
('p', 'For the user model, Shi and Gao (2014) showed analytically that the DCM (Collins et al. 2010), the between-satellite single-difference model (Ge et al. 2008; Elsobeiey and El-Rabbany 2014) and integer phase clocks (Laurichesse and Blot 2016) give the same position once the ambiguities are fixed. They did not compare the models with data, and the models may differ before fixing. For the products, network fractional cycle biases (FCBs) are usually judged by their own residuals or fixing rates (Li et al. 2016), and the user normally assumes that biases, orbits and clocks are consistent, although Ge et al. (2008) noted that narrow-lane biases absorb the modelling errors of the software that produced them. Finally, the adaptive Kalman filter of Guo and Zhang (2014) applies one variance factor to all observations, although code, phase and wide-lane observations of different systems have different noise.'),
('p', 'This paper addresses these points with one processing system and real data. (1) Float PPP and the DCM, SDBS and IPC models run in the same extended Kalman filter, with a variance factor for each observation group, so the user models can be compared on identical data. (2) The integer decision combines Dong–Bock ordering, partial fixing by the bootstrapped success rate, the FFRT and satellite exclusion; two optional machine-learning components target biased float ambiguities. (3) The software checks that the bias, clock and orbit files come from the same analysis centre and tests whether the float narrow-lane ambiguities cluster at integers, and FCBs estimated from a station network are validated at stations left out of the estimation. The components are evaluated with 24 h of GPS+Galileo data from IGS station MADR, products of four analysis centres, FCBs estimated from 48 European stations, and a step-by-step comparison against the previous version of the software.'),

('h1', 'Methods'),
('h2', 'Observation Model'),
('p', 'GPS is processed on L1/L2 and Galileo on E1/E5a. For receiver $r$ and satellite $s$, the ionosphere-free (IF) phase and code and the Melbourne–Wübbena (MW) combination are used. In all ambiguity-resolution models, the IF ambiguity in metres is written with the integer narrow-lane ambiguity $N_1$ and the wide-lane ambiguity $N_W = N_1 - N_2$:'),
('eq:bif', 'B_{"IF"} = %lambda_N N_1 + %beta N_W, ~~~~ %lambda_N = c over {f_1 + f_2}, ~~~~ %beta = {c f_2} over {f_1^2 - f_2^2}'),
('p', 'For GPS, $%lambda_N approx 10.7$ cm, $%beta approx 37.7$ cm and the wide-lane wavelength is 86.2 cm. The zenith standard deviations of the raw code and phase (0.3 m and 3 mm) are divided by $"max"(0.1, sin e)$ for elevation $e$ and propagated, with their correlations, to the IF and MW rows of each satellite.'),
('p', 'Satellite positions are interpolated at the emission time and rotated for the Earth rotation during the travel time, computed from the geometric distance to the approximate receiver position; computing it from the pseudorange would include the receiver clock offset. The models include the relativistic clock correction and the Shapiro delay, a Saastamoinen hydrostatic delay with Niell mapping functions and an estimated wet delay, solid Earth, ocean and pole tides following the IERS Conventions (Petit and Luzum 2010), the satellite attitude of the analysis centre (ORBEX quaternions, nominal yaw otherwise), satellite and receiver antenna offsets and variations, and phase wind-up.'),

('h2', 'Common Extended Kalman Filter'),
('p', 'All solutions use one state vector with the receiver position, the receiver clocks of each system, the residual zenith wet delay and an ambiguity block per satellite (one IF ambiguity in float PPP, the pair $(N_1, N_W)$ in PPP-AR). Each user model supplies only its measurement function; the time and measurement updates (iterated EKF, Joseph form) and the IGG-III outlier handling (Yang et al. 2002) are shared. Outliers inflate all rows of a satellite, which keeps the update consistent for the correlated rows of the single-difference model.'),
('p', 'In the adaptive variant, each observation group $g$ (code, phase and MW of each system) has its own variance factor $%alpha_g$. Because the receiver clock enters all rows of a group equally, the predicted innovations $bold v$ are centred with $bold C = bold I - bold 1 bold 1^T / m_g$, and the factor is a fading-memory moment estimate,'),
('eq:alpha', '%alpha_g = { sum from{k} (1 - %lambda)^{n-k} sum from{i in g} left( {(bold C bold v_k)}_i^2 - {(bold C bold H_k bold P_k^{"−"} bold H_k^T bold C^T)}_{ii} right) } over { sum from{k} (1 - %lambda)^{n-k} sum from{i in g} {(bold C bold R_{0,k} bold C^T)}_{ii} }'),
('p', 'with $%lambda = 0.05$ and $%alpha_g in left[ 0.2, 25 right]$; $bold R_0$ is the covariance before scaling. Predicted rather than post-fit residuals are used, because post-fit residuals depend on the factor and drive it to its lower limit.'),

('h2', 'User Models and Product Corrections'),
('p', 'In PPP-AR each satellite contributes an IF code, an IF phase and an MW row. With observable-specific signal biases (OSB) $C_i^s$ and $L_i^s$ of the codes and phases, the corrections added to these rows, including the satellite clock, are'),
('eq', 'c_P^s = c dt^s - {f_1^2 C_1^s - f_2^2 C_2^s} over {f_1^2 - f_2^2}, ~~~~ c_L^s = c dt^s - {f_1^2 L_1^s - f_2^2 L_2^s} over {f_1^2 - f_2^2}, ~~~~ c_W^s = -left( {f_1 L_1^s - f_2 L_2^s} over {f_1 - f_2} - {f_1 C_1^s + f_2 C_2^s} over {f_1 + f_2} right)'),
('p', 'and FCB products are converted in the same way. The three user models differ in how they remove the receiver biases (Shi and Gao 2014). The DCM estimates a code, a phase and a wide-lane clock per system as white noise and fixes the ambiguities of a reference satellite to zero. SDBS differences each satellite with the reference satellite of its system, $bold y_{"SD"} = (bold D otimes bold I_3) bold y$, $bold R_{"SD"} = (bold D otimes bold I_3) bold R (bold D otimes bold I_3)^T$, which removes the receiver clocks. In IPC the MW rows are not filtered; each wide lane is rounded from its arc mean $overline m^s$ after removing a receiver wide-lane bias computed as a circular mean, so that fractions near ±0.5 cycle do not cancel:'),
('eq', 'b_{r,W} = 1 over {2 %pi} "arg" left( sum from{s} e^{2 %pi i overline m^s} right), ~~~~ N_W^s = "round" left( overline m^s - b_{r,W} right)'),
('p', 'When the reference satellite changes, the ambiguities of the system are shifted by an integer through a linear transformation of the state and covariance, so converged and fixed arcs are kept. Before processing, the analysis centre of each product is read from its file header (SP3 agency, clock-file analysis centre, Bias-SINEX agency, ORBEX creator), and a phase-bias file from another centre than the clock file is rejected.'),

('h2', 'Integer Decision'),
('p', 'The wide lanes are fixed first and $N_1$ then, conditioned on them. A candidate needs an arc of at least 15 min, an elevation of at least 15° and a reference satellite covered by the product. The fraction and precision gates of PRIDE PPP-AR (Geng et al. 2019) are optional, because in a sequential filter they reject arcs that are still converging. Candidates are ordered by the probability of rounding correctly (Dong and Bock 1989), with $b = abs{ hat a - "round"(hat a) }$:'),
('eq', 'P_0 = 1 - sum from{n=1} to{infinity} left( func erfc left( {n - b} over {sqrt{2} %sigma} right) - func erfc left( {n + b} over {sqrt{2} %sigma} right) right)'),
('p', 'The integer least-squares problem is solved with LAMBDA after the decorrelating transformation $bold Q_{hat z} = bold Z^T bold Q_{hat a} bold Z = bold L^T bold D bold L$. The largest subset of decorrelated ambiguities whose bootstrapped success rate'),
('eq', 'P_{"IB"} = prod from{i} func erf left( 1 over {2 sqrt 2 %sigma_{i|I}} right)'),
('p', 'reaches 0.999 is fixed (Verhagen et al. 2011), where $%sigma_{i|I}^2$ are the diagonal elements of $bold D$. The subset is accepted if $P_{"IB"} >= 1 - P_f$ or if the ratio $%mu = q_2/q_1$ of the squared norms of the second-best and best candidates exceeds the FFRT threshold for $P_f = 0.001$. The threshold is computed at each epoch from $N_{"MC"} = 3000$ float vectors simulated from $N(0, bold Q_{hat z})$:'),
('eq', '%mu(P_f, bold Q_{hat z}) = "min" lbrace %mu^{"*"} : 1 over N_{"MC"} sum from{k=1} to{N_{"MC"}} "I" left( {check bold z}_k <> 0 and %mu_k >= %mu^{"*"} right) <= P_f rbrace'),
('p', 'If the test fails, the candidate with the lowest $P_0$ is removed and the stage repeated, at most twice and while at least four ambiguities remain. Fixed values do not enter the float filter unless fix-and-hold is selected.'),

('h2', 'Machine-Learning Components'),
('p', 'The **acceptance test** is a gradient-boosted classifier that estimates the probability that the integers of a stage are correct. Besides formal quantities (ratio, success rates, ADOP, subset size, squared norms), it uses three features that change when the float ambiguities are biased: the margin to the FFRT threshold $ln(%mu / %mu(P_f))$, the smallest $P_0$ and the largest fractional part. It is trained on simulated float solutions in which the covariance is mis-scaled by $%kappa sim exp N(0, 0.4^2)$ and, in 20% of the samples, one or two ambiguities are shifted by up to 0.25 cycle, and its probabilities are calibrated with isotonic regression; a fix is accepted above 0.99.'),
('p', 'The **fixability model** is trained without manual labels. A first run without fixing gives the float ambiguities at every epoch. If an arc ends with $%sigma < 0.05$ cycle and a fraction below 0.10 cycle, its integer $a^{"*"}$ is known afterwards, with $N_1$ conditioned on the wide lane:'),
('eq:cond', '{tilde N}_1 = {hat N}_1 + {%sigma_{N_1 N_W}} over {%sigma_{N_W}^2} ( N_W^{"*"} - {hat N}_W )'),
('p', 'Each earlier epoch is labelled by whether rounding gave $a^{"*"}$, and a gradient-boosted model predicts this from the standard deviation, fraction, $P_0$, elevation and arc length, cross-fitted by satellite. In the second run its probability replaces $P_0$ in the ordering and candidates below 0.5 are dropped.'),

('h2', 'Network FCB Estimation'),
('p', 'Each network station is processed with the float filter and the same products and models, the second time with its coordinate held. The float ambiguity of station $i$ and satellite $j$ in cycles is modelled as $b_{ij} = n_{ij} + r_i + s_j + %varepsilon_{ij}$, with integer $n_{ij}$ and receiver and satellite biases $r_i$ and $s_j$. Wide lanes are arc means of the MW combination (Ge et al. 2008); after they are fixed, the narrow-lane value at the end of each 15-min block is $b_{ij}^{"NL"} = ( B_{"IF",ij} - %beta N_{W,ij} ) / %lambda_N$, used when its standard deviation is below 0.2 cycle. Three estimators are compared: a batch adjustment per block (Li et al. 2016), a Kalman filter with random-walk biases, and the same filter with process noise and observation weights learned from a first run. An FCB is kept only if enough stations contribute to it. Each estimator is validated by leaving stations out: the FCBs are estimated without station $k$ and applied to its ambiguities, and the RMS and the share within ±0.15 cycle of the remaining fractions are reported.'),

('h2', 'Narrow-Lane Product Test'),
('p', 'The test holds the receiver at a known coordinate, runs the AR filter without fixing, and collects every 15 min the fractions of ${tilde N}_1$ (Eq. {eq:cond}) of arcs longer than 30 min with ${tilde %sigma}_{N_1} < 0.05$ cycle. For uniformly distributed fractions, 30% lie within ±0.15 cycle; with consistent products most lie within ±0.10 cycle (Ge et al. 2008). Running the test with and without the physical models separates model errors from product errors.'),

('h1', 'Data and Experimental Design'),
('p', 'The user station is MADR (Madrid, Spain; JAVAD TRE_G3TH receiver, AOAD/M_T antenna). Its 30-s RINEX 3 file of November 23, 2023 (day 327) was processed with GPS (C1W/L1W, C2W/L2W) and Galileo (C1X/L1X, C5X/L5X) in static mode with elevation masks of 10° (filter) and 15° (AR). The reference coordinate is the IGS daily combined SINEX solution. Four MGEX product sets were used, each with the orbits, clocks, OSB, attitude and Earth rotation parameters of one centre (Table {tab:products}); GRG provides no ERP file, so the CODE file was used. Antenna corrections are from igs20_2287.atx and ocean loading from the FES2004 BLQ table distributed with Net_Diff (Zhang 2024). For the network experiment, 50 IGS stations within about 2,500 km of MADR were downloaded (Fig. {fig:map}); two RINEX 4 files could not be read, so 48 stations (46 with Galileo) were processed with the COD products. MADR is not part of the network. The map was drawn with PyGMT (Tian et al. 2025).'),
('tab:products', 'Product sets (November 23, 2023). G/E: GPS and Galileo satellites with phase biases.',
 ['Set', 'Product line', 'Orbit / clock interval', 'Phase biases (G/E)', 'Galileo phase signals', 'Note'],
 [['COD', 'COD0MGXFIN', '5 min / 30 s', '33/25', 'L1C, L1X, L5Q, L5X', 'Final'],
  ['GRG', 'GRG0MGXFIN', '5 min / 30 s', '32/23', 'L1C, L1X, L5Q, L5X', 'Final; ERP of CODE'],
  ['WUM', 'WUM0MGXFIN', '5 min / 30 s', '31/23', 'L1C, L1X, L5Q, L5X', 'Final'],
  ['GFZ', 'GFZ0MGXRAP', '5 min / 30 s', '31/23', 'L1C, L5C', 'Rapid; no bias for X signals']]),
('fig:map', str(FIG/'station_map.png'), 'User station MADR (star) and the IGS stations of the FCB network; grey stations (RINEX 4) were not processed. Circles: 500, 1,000 and 2,000 km from MADR.', 15.5),
('p', 'The day was divided into twelve independent 2-h sessions, with the filter restarted at the start of each. For each session we computed the convergence time (horizontal and vertical errors below 10 cm for 20 consecutive epochs), the time to first fix (TTFF; fixed for 20 consecutive epochs), the fix rate, the wrong-fix rate (fixed epochs after 30 min with a 3D error above 10 cm) and the RMS of the errors in the last 30 min. Unconverged sessions were assigned 120 min. Configurations were compared with the session-wise differences using the Wilcoxon signed-rank test (Wilcoxon 1945) and a percentile bootstrap with 5,000 resamples for the 95% confidence interval of the median difference (Efron and Tibshirani 1993). Because the sessions share the satellite geometry and products of one day, the tests show whether a difference is consistent across sessions, not how it generalizes to other days.'),

('h1', 'Results'),
('h2', 'Narrow-Lane Product Test'),
('p', 'With the COD, GRG and WUM sets, 96–98% of the narrow-lane fractions lie within ±0.15 cycle and the median absolute fraction is about 0.03 cycle (Table {tab:nl}, Fig. {fig:nl}). The GFZ rapid set gives 50%, with many fractions near ±0.5 cycle. GFZ provides Galileo phase biases only for the C signals, which MADR does not track, so only GPS was tested. As a negative control, the GRG biases were combined with the GFZ orbits and clocks: the header check rejects this set, and with the check bypassed only 64% of the fractions lie within ±0.15 cycle. The physical models raised the share within ±0.10 cycle by 4–9 percentage points for the consistent sets.'),
('tab:nl', 'Narrow-lane product test at MADR: arcs, median absolute fraction and share of fractions within ±0.10 and ±0.15 cycle; last column without the physical models.',
 ['Product set', 'Arcs', 'Median |fraction| (cycle)', '±0.10 (%)', '±0.15 (%)', '±0.10 without models (%)'], nl_rows()),
('fig:nl', str(FIG/'nl_test.png'), 'Narrow-lane fractions at MADR for each product set with all models (blue) and without the physical models (red); dashed line: uniform distribution.', 16),

('h2', 'Experiment A: Analysis-Centre Products'),
('p', 'Table {tab:prodperf} summarizes the sessions. With COD, GRG and WUM, all sessions converged. Float PPP reached 10 cm after a median of 19.5–29.5 min, and its horizontal RMS in the last 30 min was 3.2–3.5 cm, mostly in the east component (Fig. {fig:prod}). SDBS fixed after 15 min in every set, fixed 84% of the epochs, and reduced the horizontal RMS to 0.6–0.8 cm. DCM and IPC fixed later (median TTFF 22–37 min; Fig. {fig:prodbox}) and less often (58–64%), and their accuracy lay between float PPP and SDBS. All PPP-AR techniques reduced the horizontal RMS significantly for every consistent set ($p <= 0.012$; Table {tab:prodtests}). The convergence time was shorter with SDBS, but not significantly ($p >= 0.07$), because the float solution often reaches 10 cm at about the time of the first fix. The vertical RMS changed by less than 0.4 cm.'),
('tab:prodperf', 'Experiment A: twelve 2-h sessions per product set and technique (medians for times, means for rates, RMS over sessions of the last 30 min).',
 ['Solution']+PERF, product_rows()),
('fig:prod', str(FIG/'products_sessions_cod.png'), 'Experiment A, COD products: median horizontal and vertical errors of the twelve sessions (lines) and their 25–75% range (bands); dashed line: 10 cm.', 15.5),
('fig:prodbox', str(FIG/'products_convergence_box.png'), 'Experiment A: convergence time of the twelve sessions for each product set and technique (unconverged sessions: 120 min).', 15.5),
('p', 'The GFZ set shows the effect of inconsistent products. The float solution was as accurate as with the other sets, but 63–71% of the fixed epochs were wrong and only 5–9 of the 12 sessions converged. The acceptance tests cannot detect this case, because the float ambiguities are biased by the product while their formal precision is high; the narrow-lane test detects it before any integer is fixed.'),
('tab:prodtests', 'Experiment A: Wilcoxon signed-rank tests of the session-wise differences from float PPP (negative: PPP-AR better), with 95% bootstrap interval of the median difference.',
 ['Comparison', 'Measure', 'Median difference', '95% interval', 'p'], test_rows(products_tests)),

('h2', 'Experiment B: Network FCBs'),
('p', 'In leave-one-out validation (Table {tab:loo}), the batch estimator gave the smallest residuals (RMS 0.057 cycle, 97% within ±0.15 cycle), followed by the learned-noise filter (0.075 cycle) and the plain filter (0.079 cycle). The network FCBs became available only at 03:15 GPS time, because the narrow-lane standard deviation of a network starting at 00:00 needs about 3 h to fall below 0.2 cycle; the first session was therefore excluded. In the remaining eleven sessions (Table {tab:netperf}, Fig. {fig:net}), SDBS fixed after 15 min with all estimators, but less often than with the COD products (73–78%) and with more wrong fixes (3–7%). The horizontal RMS improved by a median of 0.3–0.8 cm, significantly for the Kalman-filter FCBs with SDBS ($p = 0.024$) but not for the batch FCBs (Table {tab:nettests}). The estimator with the best leave-one-out statistics thus gave the most wrong fixes at the user station (7.0%). The batch solution estimates each block independently, so a block with few stations can produce an FCB that jumps relative to its neighbours, whereas the filter carries information between blocks; a left-out network station shares these blocks and satellites and does not reveal such jumps.'),
('tab:loo', 'Leave-one-out validation of the network FCBs: left-out stations, ambiguities, RMS of the remaining fractions and share within ±0.15 and ±0.25 cycle.',
 ['Estimator', 'System', 'Stations', 'Ambiguities', 'RMS (cycle)', '±0.15 (%)', '±0.25 (%)'], loo_rows()),
('tab:netperf', 'Experiment B: eleven 2-h sessions with network FCBs (same measures as Table {tab:prodperf}).',
 ['Solution']+PERF, network_rows()),
('fig:net', str(FIG/'network_sessions.png'), 'Experiment B: median horizontal and vertical errors for float PPP and SDBS with the FCBs of the three network estimators.', 15.5),
('tab:nettests', 'Experiment B: Wilcoxon signed-rank tests of the session-wise differences from float PPP.',
 ['Comparison', 'Measure', 'Median difference', '95% interval', 'p'], test_rows(network_tests)),

('h2', 'Contribution of Each Component'),
('p', 'The COD/SDBS solution was rebuilt step by step from the previous version of the software (S0: fixed ratio threshold of 3, no satellite exclusion, none of the physical models added here). Each step adds one component and is tested against the step it was added to (Tables {tab:ablation} and {tab:increments}, Figs. {fig:ablation} and {fig:ablsessions}).'),
('p', 'The decision chain (S2) caused the largest change. With the ratio test (S0, S1), integers were accepted in only 1% of the epochs, and in one S0 session 96% of the fixed epochs were wrong. With the decision chain, the fix rate rose to 84%, the TTFF was 15 min in every session, and the horizontal RMS fell by a median of 1.7 cm ($p < 0.001$). The physical models (S1) did not change the fix rate, because the ratio test remained the bottleneck, but they reduced the vertical RMS from 2.8 to 1.9 cm and moved the narrow-lane fractions towards integers. Adding the PRIDE gates (S2g) delayed the first fix from 15 to 53 min and lowered the fix rate to 33%: in a filter, the fractions of converging arcs are still large, so the gates remove arcs that the success rate and the FFRT would have fixed correctly.'),
('tab:ablation', 'Contribution of each component (COD products, SDBS, twelve 2-h sessions; same measures as Table {tab:prodperf}).',
 ['Step']+PERF, ablation_rows()),
('tab:increments', 'Change caused by each component relative to the step it was added to: median of the session-wise differences with Wilcoxon p value in parentheses. Sessions without a fix enter the TTFF with 120 min.',
 ['Step − base', 'TTFF (min)', 'Fix (%)', 'Wrong fixes (%)', 'Hor. RMS (cm)'], increment_rows()),
('fig:ablation', str(FIG/'ablation_box.png'), 'Time to first fix, fix rate and wrong-fix rate of the twelve sessions for each step of Table {tab:ablation}.', 15.5),
('p', 'The group variance factors (S3) did not change the TTFF and lowered the fix rate by less than 1 percentage point; at a static station with one receiver type, fixed weights were already adequate. The ML acceptance test (S4) gave the lowest mean wrong-fix rate (0.7% versus 1.4%), at the cost of a later first fix (+2.5 min, $p = 0.016$) and a 5-percentage-point lower fix rate; wrong fixes were too rare for the difference in wrong fixes to be significant. The fixability model (S5) delayed the first fix by a median of 20 min and lowered the fix rate by 14 percentage points ($p = 0.002$). In 2-h sessions few arcs reach the label criterion, so the model is trained on few examples; in the 24-h solution its fix rate (90%) was close to that of S2 (93%). Fix-and-hold (S6) did not change the medians, but in two sessions a wrong fix was held and the solution did not converge.'),
('fig:ablsessions', str(FIG/'ablation_sessions.png'), 'Median horizontal and vertical errors of the twelve sessions for float PPP, the previous version (S0), the decision chain (S2), the ML acceptance test (S4) and the fixability model (S5).', 15.5),

('h1', 'Conclusions'),
('p', 'A GPS+Galileo PPP-AR software was presented in which float PPP and the DCM, SDBS and IPC user models share one Kalman filter and one integer decision, and it was evaluated with data from IGS station MADR. With consistent analysis-centre products, PPP-AR reduced the horizontal RMS of 2-h static sessions from 3.2–3.5 to 0.6–0.8 cm, significantly for every product set and user model; SDBS fixed earliest and most often. Inconsistent products produced mostly wrong fixes that the acceptance tests could not detect, whereas the narrow-lane product test, which needs only a known coordinate, identified them beforehand. FCBs estimated from 48 stations improved the accuracy less than the analysis-centre products, and the estimator with the best leave-one-out statistics was not the best at the user station, so FCB estimators should also be validated at an independent user station. Most of the improvement over the previous version came from the decision chain and the physical models; the ML acceptance test halved the mean wrong-fix rate at a small cost in availability, whereas strict candidate gates, the fixability model in short sessions and fix-and-hold should not be used by default. The results come from one static station and one day; further work will cover kinematic data, multi-day series and a network processed continuously over several days.'),

('h1', 'Data Availability Statement'),
('p', 'Some or all data, models, or code generated or used during the study are available in a repository online in accordance with funder data retention policies (the PPP-AR software, the processing and analysis scripts, and the instructions to reproduce the experiments: https://github.com/fkarlitepe/GPS-Processing).'),
('p', 'Some or all data, models, or code used during the study were provided by a third party (GNSS observations of the IGS stations; MGEX orbit, clock, bias and attitude products of CODE, CNES/CLS, Wuhan University and GFZ; IGS SINEX coordinates and antenna calibrations). Direct requests for these materials may be made to the provider as indicated in the Acknowledgments.'),
('p', 'Some or all data, models, or code that support the findings of this study are available from the corresponding author upon reasonable request (the estimated fractional cycle biases and the session results).'),

('h1', 'Acknowledgments'),
('p', 'The author thanks the International GNSS Service (IGS, https://igs.org) and its Multi-GNSS Experiment (MGEX) for the GNSS observations, SINEX solutions and antenna calibrations, and the analysis centres CODE, CNES/CLS, Wuhan University and GFZ for the orbit, clock and bias products. These data are available from the IGS global data centres.'),

]

REFERENCES = [
('ref', 'Elsobeiey, M., and A. El-Rabbany. 2014. “Efficient between-satellite single-difference precise point positioning model.” *J. Surv. Eng.* 140 (2): 04014007. https://doi.org/10.1061/(ASCE)SU.1943-5428.0000125.'),
('ref', 'Grinter, T., C. Roberts, and V. Janssen. 2020. “Ambiguity-resolved real-time precise point positioning as a potential fill-in service for sparse CORS networks.” *J. Surv. Eng.* 146 (2): 04020007. https://doi.org/10.1061/(ASCE)SU.1943-5428.0000314.'),
('ref', 'Konakoglu, B., and V. Yilmaz. 2024. “Evaluating the performance of the static PPP-AR in a forest environment.” *J. Surv. Eng.* 150 (2): 05023006. https://doi.org/10.1061/JSUED2.SUENG-1403.'),
('ref', 'Ma, H., S. Verhagen, D. Psychas, J. F. G. Monico, and H. A. Marques. 2021. “Flight-test evaluation of integer ambiguity resolution enabled PPP.” *J. Surv. Eng.* 147 (3): 04021013. https://doi.org/10.1061/(ASCE)SU.1943-5428.0000367.'),
('ref', 'Martín, A., A. B. Anquela, R. Capilla, and J. L. Berné. 2011. “PPP technique analysis based on time convergence, repeatability, IGS products, different software processing, and GPS+GLONASS constellation.” *J. Surv. Eng.* 137 (3): 99–108. https://doi.org/10.1061/(ASCE)SU.1943-5428.0000047.'),
('ref', 'Wang, S., B. Li, H. Ge, and Z. Zhang. 2020. “Algorithm and assessment of ambiguity-fixed PPP with BeiDou observations and regional network augmentation.” *J. Surv. Eng.* 146 (2): 04020009. https://doi.org/10.1061/(ASCE)SU.1943-5428.0000306.'),
('ref', 'Collins, P., S. Bisnath, F. Lahaye, and P. Héroux. 2010. “Undifferenced GPS ambiguity resolution using the decoupled clock model and ambiguity datum fixing.” *Navigation* 57 (2): 123–135.'),
('ref', 'Dong, D., and Y. Bock. 1989. “Global Positioning System network analysis with phase ambiguity resolution applied to crustal deformation studies in California.” *J. Geophys. Res.* 94 (B4): 3949–3966. https://doi.org/10.1029/JB094iB04p03949.'),
('ref', 'Efron, B., and R. J. Tibshirani. 1993. *An introduction to the bootstrap*. New York: Chapman & Hall.'),
('ref', 'Ge, M., G. Gendt, M. Rothacher, C. Shi, and J. Liu. 2008. “Resolution of GPS carrier-phase ambiguities in precise point positioning (PPP) with daily observations.” *J. Geod.* 82: 389–399. https://doi.org/10.1007/s00190-007-0187-4.'),
('ref', 'Geng, J., X. Chen, Y. Pan, S. Mao, C. Li, J. Zhou, and K. Zhang. 2019. “PRIDE PPP-AR: An open-source software for GPS PPP ambiguity resolution.” *GPS Solutions* 23: 91. https://doi.org/10.1007/s10291-019-0888-1.'),
('ref', 'Guo, F., and X. Zhang. 2014. “Adaptive robust Kalman filtering for precise point positioning.” *Meas. Sci. Technol.* 25 (10): 105011. https://doi.org/10.1088/0957-0233/25/10/105011.'),
('ref', 'Laurichesse, D., and A. Blot. 2016. “Fast PPP convergence using multi-constellation and triple-frequency ambiguity resolution.” In *Proc., 29th Int. Technical Meeting of the Satellite Division of the Institute of Navigation (ION GNSS+ 2016)*. Manassas, VA: Institute of Navigation.'),
('ref', 'Li, P., X. Zhang, X. Ren, X. Zuo, and Y. Pan. 2016. “Generating GPS satellite fractional cycle bias for ambiguity-fixed precise point positioning.” *GPS Solutions* 20: 771–782. https://doi.org/10.1007/s10291-015-0483-z.'),
('ref', 'Petit, G., and B. Luzum, eds. 2010. *IERS conventions (2010)*. IERS Technical Note No. 36. Frankfurt am Main, Germany: Verlag des Bundesamts für Kartographie und Geodäsie.'),
('ref', 'Shi, J., and Y. Gao. 2014. “A comparison of three PPP integer ambiguity resolution methods.” *GPS Solutions* 18: 519–528. https://doi.org/10.1007/s10291-013-0348-2.'),
('ref', 'Teunissen, P. J. G. 1995. “The least-squares ambiguity decorrelation adjustment: A method for fast GPS integer ambiguity estimation.” *J. Geod.* 70: 65–82. https://doi.org/10.1007/BF00863419.'),
('ref', 'Tian, D., L. Uieda, W. J. Leong, et al. 2025. *PyGMT: A Python interface for the Generic Mapping Tools*. Version 0.17.0. Zenodo. https://doi.org/10.5281/zenodo.3781524.'),
('ref', 'Verhagen, S., P. J. G. Teunissen, H. van der Marel, and B. Li. 2011. “GNSS ambiguity resolution: Which subset to fix?” In *Proc., IGNSS Symp. 2011*. Sydney, Australia: Univ. of New South Wales.'),
('ref', 'Wilcoxon, F. 1945. “Individual comparisons by ranking methods.” *Biom. Bull.* 1 (6): 80–83. https://doi.org/10.2307/3001968.'),
('ref', 'Yang, Y., L. Song, and T. Xu. 2002. “Robust estimator for correlated observations based on bifactor equivalent weights.” *J. Geod.* 76: 353–358.'),
('ref', 'Zhang, Y. 2024. “Net_Diff: GNSS data processing software.” Accessed November 1, 2026. https://github.com/YizeZhang/Net_Diff.'),
]
