"""Monte Carlo training data for learned ambiguity acceptance tests.

Float ambiguity covariances come from the PPP-AR user model itself rather than random
matrices: between-satellite single differences over a random sky, either the
Melbourne-Wubbena wide-lane average (WL stage) or the ionosphere-free code/phase least
squares with position and zenith wet delay for N1 after WL fixing (N1 stage).

The float ambiguities are then drawn with errors the filter does not know about, which
is what a formal test (ratio, success rate) cannot see:
* covariance mis-scaling: true covariance = kappa * Q with log(kappa) ~ N(0, 0.4),
* unmodelled fractional biases (FCB/UPD residuals, multipath; |b| <= 0.25 cycles) on one or
  two ambiguities in 20% of the samples.
Labels compare the fixed integer combinations with the true ones, so partial fixes are
labelled correctly as well.

Besides the formal quantities (ratio, success rates, ADOP) the features hold the margin to the
fixed-failure-rate ratio threshold (FFRT, Verhagen & Teunissen 2013), the smallest Dong & Bock
P0 and the largest fractional part of the fixed ambiguities: the formal ones are blind to the
injected biases, the last two are not (bias-aware validation).
"""
import numpy as np
from gps_processing.processing.ar.integer import evaluate, ffrt_threshold, fix_probability
from gps_processing.processing.ar.models import LAMBDA_N, LAMBDA_W, combination_covariance
from gps_processing.processing.troposphere import chao_mapping

FEATURES = ['log_ratio', 'logit_success', 'logit_success_upper', 'log_adop', 'dimension', 'fixed_fraction',
            'log_best_norm', 'log_second_norm', 'stage_n1', 'log_ffrt_margin', 'logit_min_p0', 'max_fraction']


def feature_vector(result, stage='n1'):
    clip = lambda p: float(np.clip(p, 1e-9, 1-1e-9))
    logit = lambda p: np.log(clip(p)/(1-clip(p)))
    k = result['dimension']
    return np.array([np.log(max(result['ratio'], 1.0)), logit(result['success']),
                     logit(result.get('success_upper', result['success'])), np.log(result['adop']), k,
                     k/max(result.get('float_dimension', k), 1), np.log1p(result['scores'][0]/k),
                     np.log1p(result['scores'][1]/k), float(stage == 'n1'),
                     np.log(max(result['ratio'], 1e-6)/max(result.get('ffrt_threshold', 1.0), 1e-6)),
                     logit(result.get('min_p0', 1.0)), result.get('max_fraction', 0.0)])


def add_policy_features(result, a, q, failure_rate=0.001):
    """FFRT threshold, smallest P0 and largest fraction of a simulated float vector (as in policy.run_stage)."""
    combos = result['combinations']
    result['ffrt_threshold'] = ffrt_threshold(combos@q@combos.T, failure_rate, samples=1000)
    result['min_p0'] = float(fix_probability(a, np.sqrt(np.diag(q))).min())
    result['max_fraction'] = float(np.max(np.abs(a-np.rint(a))))
    return result


def sky(rng, satellites):
    elevation = np.radians(rng.uniform(10, 85, satellites))
    azimuth = rng.uniform(0, 2*np.pi, satellites)
    return elevation, azimuth


def wide_lane_covariance(rng, satellites, epochs, code, phase):
    elevation, _ = sky(rng, satellites)
    variance = np.array([combination_covariance(np.sin(e), code, phase)[2, 2] for e in elevation])
    # Multipath makes MW errors time-correlated: fewer effective epochs than observed ones.
    effective = max(1.0, epochs/rng.uniform(1, 10))
    d = np.c_[-np.ones(satellites-1), np.eye(satellites-1)]
    return d@np.diag(variance/effective)@d.T/LAMBDA_W**2


def narrow_lane_covariance(rng, satellites, epochs, code, phase, interval=30.0):
    elevation, azimuth = sky(rng, satellites)
    rate = rng.normal(0, 1, satellites)*np.radians(0.5)/60      # rad/s, ~0.5 deg/min sky motion
    m = satellites-1
    t = np.arange(epochs)[:, None]*interval
    e = np.clip(elevation+rate*t, np.radians(5), np.radians(89))  # (epochs, satellites)
    a = azimuth+rate*t
    unit = np.stack([np.cos(e)*np.sin(a), np.cos(e)*np.cos(a), np.sin(e)], -1)
    _, wet = chao_mapping(np.degrees(e))
    # Per satellite: rows Pc, Lc; unknowns dX(3), ZWD, SD N1 of satellites 1..m (0 is the reference).
    rows = np.zeros((epochs, satellites, 2, 4+m))
    rows[..., :3] = -unit[:, :, None, :]
    rows[..., 3] = wet[:, :, None]
    rows[:, 1:, 1, 4:] = np.eye(m)*LAMBDA_N
    base = combination_covariance(1.0, code, phase)[:2, :2]
    d = np.kron(np.c_[-np.ones(m), np.eye(m)], np.eye(2))
    normal = np.zeros((4+m, 4+m))
    for k in range(epochs):
        h = d@rows[k].reshape(2*satellites, 4+m)
        r = d@np.kron(np.diag(1/np.sin(e[k])**2), base)@d.T
        normal += h.T@np.linalg.solve(r, h)
    normal[:4, :4] += np.diag([1e-6]*3+[1/0.1**2])                # weak position prior, ZWD 10 cm
    return np.linalg.inv(normal)[4:, 4:]


def sample(rng, method='lambda', bias_probability=0.2):
    """One float ambiguity vector with its evaluation, feature vector and correctness label."""
    stage = 'wl' if rng.random() < 0.5 else 'n1'
    satellites = int(rng.integers(5, 11))
    epochs = int(np.exp(rng.uniform(0, np.log(120))))
    code, phase = rng.uniform(0.2, 1.0), rng.uniform(0.002, 0.005)
    q = (wide_lane_covariance if stage == 'wl' else narrow_lane_covariance)(rng, satellites, epochs, code, phase)
    q = (q+q.T)/2
    n = len(q)
    truth = rng.integers(-30, 31, n).astype(float)
    kappa = np.exp(rng.normal(0, 0.4))
    a = truth+np.linalg.cholesky(kappa*q)@rng.normal(size=n)
    if rng.random() < bias_probability:
        hit = rng.choice(n, size=int(rng.integers(1, 3)), replace=False)
        a[hit] += rng.uniform(-0.25, 0.25, len(hit))
    result = add_policy_features(evaluate(a, q, method, min_success=rng.choice([0.0, 0.9, 0.99, 0.999])), a, q)
    correct = np.array_equal(np.rint(result['combinations']@truth), np.rint(result['integers']))
    return result, feature_vector(result, stage), float(correct)


def simulate(samples=6000, seed=0, method='lambda', bias_probability=0.2, with_results=False):
    rng = np.random.default_rng(seed)
    x, y, results = [], [], []
    while len(y) < samples:
        try:
            result, features, label = sample(rng, method, bias_probability)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
        x.append(features); y.append(label)
        if with_results:
            results.append(result)
    out = (np.array(x), np.array(y))
    return (*out, results) if with_results else out
