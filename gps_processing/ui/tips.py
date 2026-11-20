"""Mouse-over explanations of the inputs: why each one is needed and what it is used for.

FIELD_TIPS: file fields (keys of App.fields). TIPS: App attributes (LabeledValue rows, buttons, lists); a dict value
gives one text per member of a dict attribute (e.g. the satellite-system check boxes).
"""

FIELD_TIPS = {
    # 01 Data files
    'rinex': 'Required. The receiver observations (code, phase, C/N0) of the station to be positioned. '
             'Every solution is computed from this file; RINEX 2.11 and 3 are read.',
    'antex': 'Antenna phase-centre offsets and variations of the receiver and satellite antennas (e.g. igs20.atx). '
             'Without it the position is biased by up to several cm, mostly in height. Use the ANTEX that matches the '
             'orbit/clock products.',
    'output': 'Folder for the results: positions, residuals, statistics, plots and reports. Must be writable.',
    'ocean_loading': 'Ocean tide loading coefficients (BLQ) for this station from the Onsala service. The crust moves '
                     'under the weight of the ocean tides: up to several cm near the coast, a few mm inland. '
                     'Needed for mm-level static PPP.',
    'erp': 'Earth rotation parameters (IGS ERP). Used for the pole tide, the crustal motion caused by polar motion '
           '(a few mm). Optional; matters for long static sessions.',
    # 02 Processing settings: products
    'sp3_day': 'Required. Precise satellite orbits (and 15 min clocks) of the observation day. PPP needs cm-level '
               'orbits; broadcast orbits are not accurate enough. Use one analysis centre for all products.',
    'sp3_before': 'Orbits of the previous day. The orbit interpolation needs points on both sides of an epoch; '
                  'without this file the first hour of the day is less accurate.',
    'sp3_after': 'Orbits of the next day. Same purpose as the previous day: avoids interpolation errors in the '
                 'last hour of the day.',
    'clock_file': 'High-rate satellite clocks (RINEX CLK, 30 s or 5 s). Satellite clocks cannot be interpolated well '
                  'from the 15 min SP3 values; a 30 s file gives faster convergence and better accuracy.',
    'attitude': 'Satellite attitude (ORBEX quaternions) of the same analysis centre. During eclipse seasons the '
                'satellites turn differently from the nominal yaw model; the true attitude corrects the antenna '
                'offset and phase wind-up.',
    'code_biases': 'Code biases (Bias-SINEX DSB/OSB). The satellite clocks refer to particular code signals; if the '
                   'receiver tracks other codes (e.g. C1C instead of C1W) the biases bring them to the same reference. '
                   'Speeds up convergence.',
    'ar_products': 'Satellite phase biases (FCB/OSB) needed to fix the ambiguities to integers (PPP-AR). They must '
                   'come from the same analysis centre as the SP3 and CLK; otherwise the integers are wrong.',
    'ionosphere': 'Global ionosphere map (IONEX). The ionosphere-free combination removes the first-order delay; '
                  'the map is used for the remaining second-order term (mm level).',
    'troposphere': 'Station zenith delays (SINEX_TRO) from IGS. Used as a priori troposphere; with ZWD estimation '
                   'on, only the small remainder is estimated, which can speed up convergence.',
    'ml_validator': 'A saved machine-learning model that decides whether an integer fix is correct. Leave empty to '
                    'train one in memory when it is first needed.',
}

TIPS = {
    # 01 Data files
    'shapiro': 'Relativistic signal delay in the Earth\'s gravity field (about 2 cm, elevation dependent). The IGS '
               'clocks assume it is applied; keep it on.',
    'satellite_pcv': 'Nadir-dependent phase-centre variations of the satellite antennas from the ANTEX file '
                     '(mm to cm). Needs the ANTEX file.',
    'example_button': 'Fills the data fields with the files found in the Data folder (observation, ANTEX, BLQ, ERP, '
                      'atmosphere products, reference stations). Orbit, clock and AR products are not chosen '
                      'automatically.',
    # Positioning technique
    'technique': 'PPP gives a float solution from orbits and clocks only. The PPP-AR models also fix the '
                 'ambiguities to integers, which shortens convergence, but they need matching phase-bias products.',
    'system_boxes': {
        'GPS': 'Use GPS L1/L2 observations.',
        'Galileo': 'Use Galileo E1/E5a observations as well. More satellites give better geometry and faster '
                   'convergence; needs multi-GNSS (MGX) orbits and clocks.',
    },
    'mask': 'Satellites below this elevation are not used. Low satellites improve the geometry but have more noise, '
            'multipath and troposphere error. 10° is usual for PPP.',
    'clock_jumps': 'Many receivers let their clock jump by 1 ms. Repairing the jump avoids restarting every phase '
                   'ambiguity at once.',
    # Estimation
    'estimator': 'EKF: extended Kalman filter, the standard PPP estimator. Adaptive EKF: also scales the observation '
                 'noise of each group from the residuals. Sequential LS: the old v2.0 solution, kept as a baseline.',
    'dynamics': 'Static: the antenna does not move, so one position is estimated for the whole session (most accurate). '
                'Kinematic: a new position is allowed at every epoch (moving receiver).',
    'position_noise': 'How fast the position may change in kinematic mode. Too small: the solution lags behind the '
                      'motion; too large: the solution is noisy.',
    'estimate_troposphere': 'The wet part of the tropospheric delay (up to about 40 cm) cannot be modelled well and is '
                            'estimated. Needed for cm accuracy, especially in height.',
    'troposphere_noise': 'How fast the wet delay may change (random walk). Larger values follow weather changes '
                         'faster but make the height noisier.',
    'physical': 'Hydrostatic delay (Saastamoinen), mapping functions and periodic relativity. Keep on; off only for '
                'tests.',
    'mapping': 'Converts the zenith tropospheric delay to the delay at the satellite\'s elevation. Niell uses '
               'latitude, height and season; Chao uses the elevation only.',
    'windup': 'The carrier phase changes when the satellite or receiver antenna rotates (up to a few cm). Needed for '
              'PPP and required by AR products.',
    'tides': 'The solid Earth tide moves the station by up to 30 cm (mostly up and down) during the day. Always keep '
             'on for accurate positions.',
    'robust': 'Reduces the weight of observations with large residuals (outliers, multipath) instead of letting them '
              'pull the solution.',
    'robust_method': 'How the weight is reduced. IGG-III: full weight, then reduced, then zero. Huber: gently reduced. '
                     'Danish: reduced quickly towards zero.',
    'huber_k': 'Standardised residual up to which an observation keeps full weight. Smaller k reduces more '
               'observations.',
    'danish_c': 'Standardised residual up to which an observation keeps full weight; above it the weight falls '
                'exponentially.',
    'outlier_test': 'Statistical test of the residuals at each epoch. Global: checks the whole epoch and only flags '
                    'it. Baarda: finds and removes the worst observation (data snooping). Pope: same with the tau '
                    'test when the variance factor is unknown.',
    'global_alpha': 'Probability of a false alarm of the global (chi-square) test. Smaller: fewer false alarms, but '
                    'small errors are missed.',
    'test_alpha': 'Probability that a correct observation is rejected by the w-test. 0.001 is the classical choice.',
    'test_power': 'Probability that an error of the design size is detected (B-method). Together with α0 it fixes '
                  'the level of the global test.',
    'max_exclusions': 'At most this many satellites are removed per epoch, so that a bad epoch cannot remove the '
                      'whole geometry.',
    'code_sigma': 'Assumed standard deviation of a code observation at the zenith. It sets the code weight relative '
                  'to the phase.',
    'phase_sigma': 'Assumed standard deviation of a phase observation at the zenith (a few mm).',
    # Ambiguity resolution
    'integer_method': 'How the float ambiguities are turned into integers. LAMBDA gives the highest success rate.',
    'success_rate': 'Only the ambiguity subset whose predicted success rate reaches this value is fixed (partial AR). '
                    'Higher: safer but fewer fixes.',
    'ar_max_std': 'Ambiguities with a larger float standard deviation are not fixed: they are not yet precise enough.',
    'ar_min_epochs': 'An ambiguity must be tracked this many epochs before it is fixed, so that its float value has '
                     'converged.',
    'ar_min_elevation': 'Satellites below this elevation are not fixed: their ambiguities are less reliable.',
    'ar_min_arc': 'Minimum tracking time before fixing, in seconds.',
    'ar_min_ambiguities': 'A fix needs at least this many ambiguities; with fewer, a wrong fix is hard to detect.',
    'ar_max_exclusions': 'After a rejected fix, up to this many satellites are dropped and the fix is tried again.',
    'validation': 'Test that decides whether to accept the integer solution. Prevents wrong fixes, which bias the '
                  'position.',
    'ar_failure_rate': 'Allowed probability of a wrong fix for the fixed-failure-rate ratio test.',
    'ar_ratio': 'The second-best integer candidate must be this many times worse than the best one.',
    'ml_threshold': 'The machine-learning validator accepts a fix only if its predicted probability of being correct '
                    'reaches this value.',
    'ml_model': 'Type of machine-learning model trained as the validator.',
    'train_button': 'Simulates many fixes (Monte Carlo), trains the validator and saves it to Data/models.',
    'ar_hold': 'Once integers have been fixed identically for some epochs, they are held as constraints. Gives a more '
               'stable solution, but a wrong fix stays longer.',
    'ar_hold_epochs': 'Number of identical fixes before the integers are held.',
    'ar_fixability': 'Learns from a first float run which ambiguities can be fixed, and uses it to order and filter '
                     'the candidates.',
    'diagnostic_button': 'Checks whether the AR products fit the data: shows how close the float ambiguities are to '
                         'integers. Use it before a PPP-AR run with new products.',
    # Machine learning
    'ml_weighting': 'Learns the observation noise from elevation, C/N0 and arc length (from a first run) and repeats '
                    'the solution with these weights.',
    # Satellite selection
    'selection': 'Uses only a subset of the visible satellites in the filter update. Studies only: in open sky, all '
                 'satellites give the best and fastest solution.',
    'selection_count': 'Number of satellites used per epoch (at least 6 for GPS + Galileo PPP).',
    'selection_weighting': 'Weights of the DOP that the subset minimises. A weighted DOP (filter, elevation or C/N0) '
                           'prefers precise satellites and gives a more accurate solution than the classical DOP.',
    'selection_objective': 'Which DOP is minimised: PDOP (3D position), GDOP (position and clock), HDOP (horizontal) '
                           'or VDOP (height).',
    'selection_interval': 'Keeps the chosen subset for this time while all its satellites stay visible. Reduces the '
                          'selection cost; 300 to 600 s cost no accuracy.',
    'selection_hysteresis': 'A new subset replaces the current one only if its DOP is better by this fraction. Avoids '
                            'switching satellites for small gains.',
    'selection_troposphere': 'Includes the wet-delay column in the DOP geometry, as in the PPP filter.',
    'selection_state': 'Keep: ambiguities of unselected satellites stay in the filter and continue when the satellite '
                       'returns (recommended). Drop: they restart, which makes the solution much worse.',
    'selection_warm_start': 'Adds the still-visible satellites of the previous subset as an extra candidate.',
    'selection_seed': 'Start value of the random generator: the same seed gives the same random subsets.',
    'selection_analysis': 'Writes the DOP / weighted-DOP tables per epoch and satellite even when all satellites are '
                          'used.',
    # Validation reference
    'reference_row': 'Known coordinate of the station (e.g. IGS/SINEX). Needed to compute the true error and the '
                     'convergence time; if empty, the last epoch of the solution is used.',
    'header_button': 'Copies the approximate position from the RINEX header (accurate only to a few metres).',
    # FCB estimation
    'fcb': 'Reference stations used to estimate the satellite phase biases (FCB) when no AR product is loaded. At '
           'least 5 stations of the same day.',
    'fcb_buttons': 'Add or remove reference-station RINEX files.',
    'fcb_method': 'Estimator of the phase biases: batch adjustment, Kalman filter, Kalman filter with ML weights, or '
                  'all three (the best is used).',
    'fcb_interval': 'Length of the blocks in which one narrow-lane bias per satellite is estimated.',
    'fcb_hold': 'Processes the reference stations a second time with fixed daily coordinates, as in network '
                'processing. Gives cleaner biases.',
    'fcb_min_contributors': 'A satellite bias needs at least this many stations; weaker blocks are dropped.',
    'fcb_workers': 'Number of stations processed in parallel.',
    'fcb_button': 'Estimates the phase biases from the reference stations and writes them as GENFCB files.',
}
