"""Ambiguity-resolution policy of one integer stage (WL or N1): candidate gates, P0 ordering,
one acceptance decision and satellite exclusion.

1. Gates (PRIDE PPP-AR defaults, ``ar_candidate_gates``): fractional part of the float ambiguity below 0.20 cycle (WL)
   or 0.15 cycle (N1) and a marginal sigma below 0.25 cycle for the LAMBDA family; arcs are
   gated by length and elevation before this function (solver).
2. Order: Dong & Bock P0, highest first (Ge et al. 2005), then sigma.
3. Integer estimation with partial AR on the decorrelated subset (``integer.evaluate``).
4. One acceptance decision (``validation``):
     'ffrt'      fixed-failure-rate ratio test (Verhagen & Teunissen 2013): accept when the IB
                 success rate of the fixed subset is >= 1 - Pf, otherwise when the ratio
                 reaches the simulated threshold mu(Pf, Q);
     'ratio'     v2.x rule: success-rate gate + constant ratio threshold;
     'ml'        learned, calibrated probability of a correct fix (features include P0 and
                 the FFRT margin);
     'ffrt+ml' / 'ratio+ml'   both must accept.
5. When the decision fails, the candidate with the lowest P0 is excluded and the stage is
   repeated (at most ``max_exclusions`` times, Verhagen et al. 2011 / PRIDE), as long as at
   least ``min_ambiguities`` remain.
"""
import numpy as np
from .integer import evaluate, condition_linear, fix_probability, ffrt_threshold

VALIDATIONS = ('ffrt', 'ratio', 'ml', 'ratio+ml', 'ffrt+ml')
FRACTION_GATE = {'wl': 0.20, 'n1': 0.15}
CANDIDATE_SIGMA = 0.25          # cycles; the success rate of the decorrelated subset does the rest


def decide(result, q, settings, validator, stage):
    """(accepted, reason); adds 'ffrt_threshold' / 'probability' to ``result``."""
    rules = settings.validation.split('+')
    if 'ratio' in rules:
        if settings.method != 'ils' and result['success'] < settings.success:
            return False, f'{stage}_success_rate_gate'
        if result['ratio'] < settings.ratio:
            return False, f'{stage}_ratio_rejected'
    if 'ffrt' in rules or 'ml' in rules:
        qz = result['combinations']@q@result['combinations'].T
        result['ffrt_threshold'] = ffrt_threshold(qz, settings.failure_rate)
        if 'ffrt' in rules and result['ratio'] < result['ffrt_threshold']:
            return False, f'{stage}_ffrt_rejected'
    if 'ml' in rules:
        result['probability'] = validator.probability(result, stage)
        if result['probability'] < settings.threshold:
            return False, f'{stage}_ml_rejected'
    return True, ''


def run_stage(state, cov, indices, settings, validator, stage, context=None):
    """Fix one stage. Returns ((fixed state, covariance) or None, result, reason, used indices).

    ``context``: (sats, elevations, arc epochs) of ``indices`` for the learned fixability model
    (``settings.fixability``), which then orders and gates the candidates instead of P0."""
    indices = list(indices)
    if not indices:
        return None, None, f'{stage}_no_candidates', []
    a = state[indices]
    sigma = np.sqrt(np.maximum(np.diag(cov)[indices], 0))
    p0 = fix_probability(a, sigma)
    settings_p0 = {indices[i]: float(p0[i]) for i in range(len(indices))}
    if settings.method == 'ils':
        # v2.0 baseline: the solver passes the four most precise candidates; precision gate, no exclusion.
        if sigma.max() >= settings.max_std:
            return None, None, f'{stage}_precision_gate', indices
        chosen = indices
    else:
        fraction = np.abs(a-np.rint(a))
        keep = (fraction < FRACTION_GATE[stage]) & (sigma < CANDIDATE_SIGMA) if settings.gates else np.ones(len(a), bool)
        score = p0
        model = getattr(settings, 'fixability', None)
        if model is not None and context is not None:
            from gps_processing.processing.ml.fixability import features
            sats, elevations, arcs = context
            score = model.probability(sats, features(a, sigma, elevations, arcs, stage == 'n1'))
            keep &= score >= settings.fixability_threshold
        order = sorted(np.flatnonzero(keep), key=lambda i: (-score[i], sigma[i]))
        chosen = [indices[i] for i in order]
        if model is not None and context is not None and len(chosen) < settings.min_ambiguities:
            # Learned search trigger: wait until enough arcs are predicted fixable.
            return None, None, f'{stage}_ml_trigger_wait', chosen
    if len(chosen) < settings.min_ambiguities:
        return None, None, f'{stage}_too_few_candidates', chosen
    result, reason = None, ''
    for attempt in range(settings.max_exclusions+1 if settings.method != 'ils' else 1):
        q = cov[np.ix_(chosen, chosen)]
        result = evaluate(state[chosen], q, settings.method, settings.success)
        result.update(min_p0=float(min(settings_p0[i] for i in chosen)), excluded=attempt,
                      max_fraction=float(np.max(np.abs(state[chosen]-np.rint(state[chosen])))))
        accepted, reason = decide(result, q, settings, validator, stage)
        if accepted:
            fixed, covariance = condition_linear(state, cov, chosen, result['combinations'], result['integers'])
            return (fixed, covariance), result, '', chosen
        if len(chosen)-1 < settings.min_ambiguities:
            break
        chosen = chosen[:-1]                    # drop the lowest-P0 candidate and retry
    return None, result, reason, chosen
