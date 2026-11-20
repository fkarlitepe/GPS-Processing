"""Statistical quality control of an adjustment / Kalman filter update.

Outlier tests (detection - identification - adaptation, DIA):
  'global'  overall model test: T = e^T S^-1 e ~ chi2(df) under H0 (innovations e, covariance S; in a
            least-squares adjustment T = v^T Qll^-1 v with df = redundancy). Detection only: the epoch is flagged.
  'baarda'  data snooping (Baarda 1968): w_i = (W v)_i / sqrt((W Qv W)_i,i) ~ N(0, 1) with the a-priori
            variance factor known. B-method: the local level alpha0 and the power 1 - beta0 fix the
            non-centrality lambda0 = (z(1-alpha0/2) + z(1-beta0))^2, and the global test gets the level at which
            it detects the same bias with the same power. Local tests run only when the global test rejects;
            the observation group (satellite) of the largest |w| > z(1-alpha0/2) is excluded and the update
            repeated (one at a time).
  'pope'    tau test (Pope 1976): the variance factor is estimated, sigma0^2 = T / df, tau_i = w_i / sigma0
            follows the tau distribution with df degrees of freedom; critical value
            tau_c = sqrt(df) t / sqrt(df - 1 + t^2), t = t(df-1, 1 - alpha_i/2), alpha_i = 1-(1-alpha)^(1/n) (Sidak).
Robust estimation (iteratively re-weighted, standardized residuals u):
  'igg3'    IGG-III (Yang 1994), k0 = 2.5, k1 = 6
  'huber'   Huber (1964): w = 1 for |u| <= k, k / |u| otherwise (k = 1.5)
  'danish'  Danish method (Krarup, Juhl & Kubik 1980), exponential form: w = 1 for |u| <= c,
            exp(1 - (u / c)^2) otherwise (c = 2); continuous at |u| = c, about 0.05 at 2c.
All weight functions return the variance inflation 1 / w (>= 1) used by the filter.

In the filter the test quantities come from the post-fit residuals of the update, which avoids solving with
the innovation covariance S (a receiver clock reset to 1e10 m^2 makes S ill-conditioned):
    v = (I - H K) e = R S^-1 e,    Qv = R - H P+ H^T = R S^-1 R,    W = R^-1  ->  W v = S^-1 e,  W Qv W = S^-1.
Degrees of freedom: rows of the non-excluded groups minus the states with a diffuse prior (variance
>= 1e4 m^2: reset receiver clocks, new ambiguities), whose rows carry no redundancy.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import stats

TESTS = ('none', 'global', 'baarda', 'pope')
ROBUST = ('igg3', 'huber', 'danish')
EXCLUDED = 1e8               # variance inflation of an excluded group (weight 1e-8, as IGG-III rejection)
DIFFUSE = 1e4                # m^2: prior variance treated as no information


@dataclass
class QualityOptions:
    test: str = 'none'            # none / global / baarda / pope
    alpha: float = 0.001          # local level (Baarda alpha0); Pope: global level split over the observations
    power: float = 0.80           # Baarda B-method power 1 - beta0
    global_alpha: float = 0.01    # level of the global test for 'global' and 'pope'
    max_exclusions: int = 3       # groups (satellites) excluded per epoch at most
    robust: str = 'igg3'          # weight function of the robust update
    huber_k: float = 1.5
    danish_c: float = 2.0

    @classmethod
    def from_config(cls, config):
        return cls(getattr(config, 'outlier_test', 'none'), getattr(config, 'test_alpha', 0.001),
                   getattr(config, 'test_power', 0.80), getattr(config, 'global_alpha', 0.01),
                   getattr(config, 'max_exclusions', 3), getattr(config, 'robust_method', 'igg3'),
                   getattr(config, 'huber_k', 1.5), getattr(config, 'danish_c', 2.0))

    @property
    def active(self):
        return self.test != 'none'


# --- critical values -------------------------------------------------------------------------------------
def baarda_lambda(alpha0=0.001, power=0.80):
    """Non-centrality lambda0 of the B-method (alpha0 two-sided, power 1 - beta0); 17.07 for 0.001 / 0.80."""
    return float((stats.norm.ppf(1-alpha0/2)+stats.norm.ppf(power))**2)


def b_method_alpha(df, alpha0=0.001, power=0.80):
    """Global-test level that detects the bias of lambda0 with the same power as the local w-test."""
    critical = stats.ncx2.ppf(1-power, df, baarda_lambda(alpha0, power))
    return float(stats.chi2.sf(critical, df))


def chi2_critical(df, alpha):
    return float(stats.chi2.ppf(1-alpha, df))


def tau_critical(df, alpha):
    """Critical value of the tau distribution (Pope 1976) with df degrees of freedom, two-sided level alpha."""
    if df <= 1:
        return np.inf
    t = stats.t.ppf(1-alpha/2, df-1)
    return float(np.sqrt(df)*t/np.sqrt(df-1+t**2))


# --- robust weight functions (variance inflation = 1 / weight) -------------------------------------------
def igg3(standardized, k0=2.5, k1=6.0):
    """IGG-III equivalent-weight factor (Yang 1994); returns variance inflation >= 1."""
    v = np.abs(standardized)
    weight = np.ones_like(v, dtype=float)
    mid = (v > k0) & (v <= k1)
    weight[mid] = k0/v[mid]*((k1-v[mid])/(k1-k0))**2
    weight[v > k1] = 1e-8
    return 1/np.maximum(weight, 1e-8)


def huber(standardized, k=1.5):
    v = np.abs(np.asarray(standardized, float))
    weight = np.where(v <= k, 1.0, k/np.maximum(v, 1e-12))
    return 1/np.maximum(weight, 1e-8)


def danish(standardized, c=2.0):
    v = np.abs(np.asarray(standardized, float))
    weight = np.where(v <= c, 1.0, np.exp(1-(v/c)**2))
    return 1/np.maximum(weight, 1e-8)


def robust_inflation(method, standardized, options=None):
    options = options or QualityOptions()
    if method == 'huber':
        return huber(standardized, options.huber_k)
    if method == 'danish':
        return danish(standardized, options.danish_c)
    if method == 'igg3':
        return igg3(standardized)
    raise ValueError(f'unknown robust method {method!r}; choose from {ROBUST}')


# --- test statistics -------------------------------------------------------------------------------------
def w_statistics(residual, weight, residual_cov, min_redundancy=1e-3):
    """Baarda w of every row and its redundancy number r_i = (Qv W)_ii; untestable rows (r_i small) get w = 0."""
    wv = weight@residual
    variance = np.einsum('ij,jk,ki->i', weight, residual_cov, weight)
    redundancy = np.einsum('ij,ji->i', residual_cov, weight)
    w = np.zeros(len(residual))
    ok = (redundancy > min_redundancy) & (variance > 0)
    w[ok] = wv[ok]/np.sqrt(variance[ok])
    return w, redundancy


@dataclass
class TestResult:
    test: str = 'none'
    statistic: float = np.nan      # T (chi2)
    df: float = 0
    critical: float = np.nan       # global critical value
    passed: bool = True            # global test of the final (adapted) model
    first_passed: bool = True      # global test before any exclusion
    sigma0: float = np.nan         # a-posteriori sqrt(T / df)
    local_max: float = np.nan      # largest |w| (Baarda) or |tau| (Pope) in the final model
    local_critical: float = np.nan
    excluded: list = field(default_factory=list)   # excluded groups, in the order of identification

    def as_dict(self, prefix='qc_'):
        return {f'{prefix}test': self.test, f'{prefix}T': self.statistic, f'{prefix}df': self.df,
                f'{prefix}critical': self.critical, f'{prefix}passed': self.passed, f'{prefix}first_passed': self.first_passed,
                f'{prefix}sigma0': self.sigma0, f'{prefix}local_max': self.local_max,
                f'{prefix}local_critical': self.local_critical, f'{prefix}excluded': len(self.excluded)}


def evaluate(options, residual, weight, residual_cov, statistic, df, rows):
    """Global and local statistics of one adapted model; ``rows`` are the rows that may be tested.
    Returns (global_critical, rejected, local values of the rows, local critical value)."""
    df = max(int(round(df)), 1)
    if options.test == 'baarda':
        global_crit = chi2_critical(df, b_method_alpha(df, options.alpha, options.power))
    else:
        global_crit = chi2_critical(df, options.global_alpha)
    rejected = statistic > global_crit
    w, _ = w_statistics(residual, weight, residual_cov)
    local, local_crit = np.zeros(len(residual)), np.nan
    if options.test == 'baarda':
        local, local_crit = w, float(stats.norm.ppf(1-options.alpha/2))
    elif options.test == 'pope':
        sigma0 = np.sqrt(max(statistic/df, 1e-12))
        n = max(int(np.sum(np.abs(w[rows]) > 0)), 1)
        local = w/sigma0
        local_crit = tau_critical(df, 1-(1-options.global_alpha)**(1/n))
    return global_crit, bool(rejected), np.where(np.isin(np.arange(len(residual)), rows), local, 0.0), local_crit


def identify(local, local_crit, groups, excluded, min_groups, groups_count):
    """Group (satellite) of the largest local statistic above its critical value, or None."""
    if not np.isfinite(local_crit) or groups_count-len(excluded)-1 < min_groups:
        return None
    order = np.argsort(-np.abs(local))
    for i in order:
        if abs(local[i]) <= local_crit:
            return None
        if int(groups[i]) not in excluded:
            return int(groups[i])
    return None


# --- Gauss-Markov least squares with the same tests -----------------------------------------------------
@dataclass
class Adjustment:
    x: np.ndarray
    cov: np.ndarray
    residual: np.ndarray
    sigma0: float
    redundancy: int
    weights: np.ndarray            # final observation weight factors (0: excluded)
    test: TestResult
    iterations: int = 1


def least_squares(A, l, Qll=None, options=None, robust=None, groups=None, iterations=20, tolerance=1e-8):
    """Weighted least squares l = A x + v with DIA (global / Baarda / Pope) and/or robust re-weighting.

    ``Qll``: cofactor matrix (identity if None); ``groups``: observation groups excluded together (default
    one per observation); ``robust``: None / 'igg3' / 'huber' / 'danish' (standardized residuals v_i / sigma_vi
    with the a-priori variance factor 1, or the a-posteriori one for Pope)."""
    A, l = np.atleast_2d(np.asarray(A, float)), np.asarray(l, float)
    m, u = A.shape
    Qll = np.eye(m) if Qll is None else np.asarray(Qll, float)
    groups = np.arange(m) if groups is None else np.asarray(groups, int)
    options = options or QualityOptions()
    inflation = np.ones(groups.max()+1)
    excluded, first = [], None

    def solve(infl):
        scale = np.sqrt(infl[groups])
        Q = Qll*scale[:, None]*scale[None, :]
        W = np.linalg.inv(Q)
        N = A.T@W@A
        Qxx = np.linalg.inv(N)
        x = Qxx@A.T@W@l
        v = l-A@x
        Qv = Q-A@Qxx@A.T
        return x, Qxx, v, W, Qv

    def dof():
        return int(np.sum(~np.isin(groups, excluded)))-u

    while True:
        x, Qxx, v, W, Qv = solve(inflation)
        statistic, df = float(v@W@v), dof()
        rows = np.flatnonzero(~np.isin(groups, excluded))
        if not options.active or df < 1:
            break
        crit, rejected, local, local_crit = evaluate(options, v, W, Qv, statistic, df, rows)
        if first is None:
            first = not rejected
        if not rejected or options.test == 'global' or len(excluded) >= options.max_exclusions:
            break
        g = identify(local, local_crit, groups, excluded, 1, len(inflation))
        if g is None or dof()-np.sum(groups == g) < 1:
            break
        excluded.append(g)
        inflation[g] = EXCLUDED
    count = 1
    if robust:
        for count in range(1, iterations+1):
            sigma = np.sqrt(np.maximum(np.diag(Qv), 1e-30))
            s0 = np.sqrt(max(v@W@v/max(dof(), 1), 1e-30)) if options.test == 'pope' else 1.0
            std = v/(sigma*s0)*np.sqrt(inflation[groups])
            worst = np.zeros(len(inflation))
            np.maximum.at(worst, groups, np.abs(std))
            updated = np.maximum(inflation, robust_inflation(robust, worst, options)) if robust == 'igg3' else \
                np.where(np.isin(np.arange(len(inflation)), excluded), EXCLUDED, robust_inflation(robust, worst, options))
            x_old = x
            inflation = updated
            x, Qxx, v, W, Qv = solve(inflation)
            if np.max(np.abs(x-x_old)) < tolerance:
                break
    statistic, df = float(v@W@v), dof()
    result = TestResult(options.test, statistic, df, sigma0=float(np.sqrt(statistic/df)) if df > 0 else np.nan,
                        excluded=excluded, first_passed=True if first is None else first)
    if options.active and df >= 1:
        rows = np.flatnonzero(~np.isin(groups, excluded))
        crit, rejected, local, local_crit = evaluate(options, v, W, Qv, statistic, df, rows)
        result.critical, result.passed, result.local_critical = crit, not rejected, local_crit
        result.local_max = float(np.max(np.abs(local[rows]))) if len(rows) and options.test != 'global' else np.nan
    weights = 1/inflation[groups]
    weights[weights < 1e-6] = 0.0
    return Adjustment(x, Qxx*(result.sigma0**2 if options.test == 'pope' and np.isfinite(result.sigma0) else 1.0),
                      v, result.sigma0, df, weights, result, count)
