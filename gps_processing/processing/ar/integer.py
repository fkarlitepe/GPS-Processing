"""Exact integer least squares, LAMBDA decorrelation, success rates and partial AR.

Success-rate bounds and approximations follow Ps-LAMBDA (Verhagen, Li & Teunissen 2013,
Computers & Geosciences 54, 361-376); equation numbers refer to that paper.

Integer estimators (Teunissen, Joosten & Tiberius 2002, "A comparison of TCAR, CIR and
LAMBDA GNSS ambiguity resolution", ION GPS 2002):
  'lambda'    integer least squares on LAMBDA-decorrelated ambiguities + partial AR
  'ils'       integer least squares of all candidates (v2.0 baseline)
  'bootstrap' integer bootstrapping of the LAMBDA-decorrelated ambiguities + partial AR;
              exact success rate, no search (paper, conclusion 4)
  'cascade'   TCAR/CIR principle: the pre-defined, frequency-based transformation
              (wide lane, then N1, i.e. the solver's two stages) with every satellite
              pair rounded on its own, ignoring the correlation between pairs.
"""
import math
import numpy as np

def search(a, covariance, max_nodes=100000):
    a = np.asarray(a, dtype=float)
    q = np.asarray(covariance, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError('Non-finite ambiguity estimate')
    upper = np.linalg.cholesky(np.linalg.inv(q)).T
    seed = np.rint(a).astype(int)
    candidates = {}
    def add(z):
        key = tuple(z)
        candidates[key] = float(np.sum((upper @ (a-z))**2))
    add(seed)
    for i in range(len(a)):
        for sign in (-1,1):
            z=seed.copy(); z[i]+=sign; add(z)
    best = sorted(candidates, key=candidates.get)[:2]
    radius = candidates[best[1]]
    z=seed.copy(); nodes=0
    def visit(k, cost):
        nonlocal nodes, radius, best
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError('Integer search budget exceeded; retain float')
        if k < 0:
            add(z.copy()); best=sorted(candidates,key=candidates.get)[:2]
            radius=candidates[best[1]]; return
        center=a[k]+upper[k,k+1:]@(a[k+1:]-z[k+1:])/upper[k,k]
        span=np.sqrt(max(0,radius-cost))/abs(upper[k,k])
        lo,hi=int(np.ceil(center-span)),int(np.floor(center+span))
        if hi-lo > max_nodes:
            raise RuntimeError('Integer search interval too large; retain float')
        for value in sorted(range(lo,hi+1),key=lambda v:abs(v-center)):
            newcost=cost+(upper[k,k]*(center-value))**2
            if newcost <= radius+1e-10:
                z[k]=value; visit(k-1,newcost)
    visit(len(a)-1,0.)
    scores=np.array([candidates[b] for b in best])
    return np.array(best), scores

def condition(state, covariance, indices, integers):
    cross=covariance[:,indices]
    gain=np.linalg.solve(covariance[np.ix_(indices,indices)],cross.T).T
    fixed=state+gain@(integers-state[indices])
    q=covariance-gain@cross.T
    return fixed,(q+q.T)/2

def condition_linear(state, covariance, indices, combinations, integers):
    """Condition on integer-valued linear combinations z = combinations @ state[indices]."""
    g = np.zeros((len(combinations), len(state)))
    g[:, indices] = combinations
    cross = covariance@g.T
    gain = np.linalg.solve(g@cross, cross.T).T
    fixed = state+gain@(integers-g@state)
    q = covariance-gain@cross.T
    return fixed, (q+q.T)/2

def ldl(q):
    """Q = L^T diag(D) L with unit lower-triangular L (LAMBDA ordering, De Jonge & Tiberius 1996)."""
    n = len(q); a = np.array(q, dtype=float); lower = np.zeros((n, n)); d = np.zeros(n)
    for i in range(n-1, -1, -1):
        d[i] = a[i, i]
        if d[i] <= 0:
            raise np.linalg.LinAlgError('Ambiguity covariance is not positive definite')
        lower[i, :i+1] = a[i, :i+1]/np.sqrt(d[i])
        for j in range(i):
            a[j, :j+1] -= lower[i, :j+1]*lower[i, j]
        lower[i, :i+1] /= lower[i, i]
    return lower, d

def decorrelate(q):
    """Integer Gauss transformations and permutations; returns unimodular Z, L, D of Z^T Q Z."""
    lower, d = ldl(q)
    n = len(d); z = np.eye(n)
    def gauss(i, j):
        mu = round(lower[i, j])
        if mu:
            lower[i:, j] -= mu*lower[i:, i]
            z[:, j] -= mu*z[:, i]
    j = k = n-2
    while j >= 0:
        if j <= k:
            for i in range(j+1, n):
                gauss(i, j)
        delta = d[j]+lower[j+1, j]**2*d[j+1]
        if delta+1e-6 < d[j+1]:
            eta, lam = d[j]/delta, d[j+1]*lower[j+1, j]/delta
            d[j], d[j+1] = eta*d[j+1], delta
            for col in range(j):
                a0, a1 = lower[j, col], lower[j+1, col]
                lower[j, col] = -lower[j+1, j]*a0+a1
                lower[j+1, col] = eta*a0+lam*a1
            lower[j+1, j] = lam
            lower[j+2:, [j, j+1]] = lower[j+2:, [j+1, j]]
            z[:, [j, j+1]] = z[:, [j+1, j]]
            k = j; j = n-2
        else:
            j -= 1
    return np.rint(z), lower, d

def bootstrap_success(conditional_variances):
    """Exact IB success rate, eq. (28): prod(2*Phi(1/(2*sigma_i|I)) - 1) = prod erf(1/(2*sqrt(2)*sigma)).

    With the conditional variances of the Z-decorrelated ambiguities it is the sharp lower
    bound of the ILS success rate, eq. (32)."""
    return float(np.prod([math.erf(1/(2*math.sqrt(2*v))) for v in conditional_variances]))


def bootstrap(a, covariance):
    """Integer bootstrapping estimator, eq. (15), in the LAMBDA order (last ambiguity first)."""
    lower, _ = ldl(covariance)
    a = np.asarray(a, dtype=float); n = len(a)
    conditional, fixed = np.zeros(n), np.zeros(n)
    for i in range(n-1, -1, -1):
        conditional[i] = a[i]-lower[i+1:, i]@(conditional[i+1:]-fixed[i+1:])
        fixed[i] = np.rint(conditional[i])
    return fixed


def chi2_cdf(x, dof):
    """P(chi^2(dof) <= x) from the series of the regularized lower incomplete gamma function."""
    if x <= 0:
        return 0.0
    a, y = dof/2, x/2
    term = total = 1/a
    k = 0
    while abs(term) > 1e-16*abs(total) and k < 100000:
        k += 1
        term *= y/(a+k)
        total += term
    return float(min(1.0, math.exp(-y+a*math.log(y)-math.lgamma(a))*total))


def adop_upper_bound(adop, n):
    """ADOP-based ILS upper bound, eq. (33): P(chi^2(n) <= c_n/ADOP^2), c_n = (n/2 Gamma(n/2))^(2/n)/pi."""
    c_n = (n/2*math.gamma(n/2))**(2/n)/math.pi
    return chi2_cdf(c_n/adop**2, n)


def success_rates(covariance):
    """Closed-form Ps-LAMBDA bounds/approximations for a float ambiguity covariance (cycles^2).

    ir_original     IR lower bound of the original ambiguities, eq. (26): the 'cascade' estimator
    ib_original     exact IB success rate of the original ambiguities, eq. (28)
    ib              exact IB success rate after Z-decorrelation, eqs. (29), (32): ILS lower bound
    ir_lower        IR lower bound for the decorrelated ambiguities, eq. (26)
    adop            ambiguity dilution of precision (cycles), eq. (31)
    adop_approx     (2 Phi(1/(2 ADOP)) - 1)^n: IB upper bound and ILS approximation, eq. (30)
    ils_upper       ADOP-based ILS upper bound, eq. (33)
    eig_lower/upper ILS bounds from the extreme eigenvalues of Q_zz, eqs. (36)-(37)
    """
    q = np.asarray(covariance, dtype=float); n = len(q)
    _, d_original = ldl(q)
    z, _, d = decorrelate(q)
    qz = z.T@q@z
    adop = float(np.exp(np.mean(np.log(d))/2))
    eig = np.linalg.eigvalsh(qz)
    one = lambda variance: math.erf(1/(2*math.sqrt(2*variance)))
    return dict(ir_original=float(np.prod([one(v) for v in np.diag(q)])),
                ib_original=bootstrap_success(d_original), ib=bootstrap_success(d),
                ir_lower=float(np.prod([one(v) for v in np.diag(qz)])), adop=adop,
                adop_approx=one(adop**2)**n, ils_upper=adop_upper_bound(adop, n),
                eig_lower=one(eig.max())**n, eig_upper=one(eig.min())**n)


def simulate_success(covariance, method='ils', samples=100000, seed=0):
    """Simulation-based success rate, eq. (25): fraction of N(0, Q) draws mapped to 0.

    All three estimators run on the decorrelated ambiguities (ILS is Z-invariant; IR and IB
    are not, and decorrelation is their favourable parameterization). Ps-LAMBDA uses >= 1e5."""
    q = np.asarray(covariance, dtype=float)
    z, _, _ = decorrelate(q)
    qz = z.T@q@z
    draws = np.random.default_rng(seed).standard_normal((samples, len(q)))@np.linalg.cholesky(qz).T
    if method == 'ir':
        hits = int(np.all(np.rint(draws) == 0, axis=1).sum())
    elif method == 'ib':
        hits = sum(not bootstrap(x, qz).any() for x in draws)
    else:
        hits = sum(not search(x, qz)[0][0].any() for x in draws)
    return hits/samples


INTEGER_METHODS = ('lambda', 'ils', 'bootstrap', 'cascade')


def runner_up(z_hat, q, best):
    """Best of the 2k unit neighbours of ``best``: the second candidate for the ratio of
    estimators without a search (bootstrapping, rounding)."""
    inverse = np.linalg.inv(q)
    norm = lambda c: float((z_hat-c)@inverse@(z_hat-c))
    neighbours = [best+sign*np.eye(len(best))[i] for i in range(len(best)) for sign in (-1, 1)]
    second = min(neighbours, key=norm)
    return np.array([best, second]), np.array([norm(best), norm(second)])


def precise_prefix(probabilities, min_success):
    """Largest k such that the product of the first k probabilities reaches ``min_success`` (k >= 1)."""
    k = 0
    while k < len(probabilities) and np.prod(probabilities[:k+1]) >= min_success:
        k += 1
    return max(k, 1)


def evaluate(a, covariance, method='lambda', min_success=0.999):
    """Float -> integer candidates for one AR stage.

    Returns the fixed integer combinations (``combinations @ a`` is integer), the best and
    second candidate with their squared norms, ``ratio``, ``success`` and ADOP.

    lambda    LAMBDA decorrelation, partial AR on the largest decorrelated subset whose
              bootstrapped success rate reaches ``min_success`` (Teunissen et al. 1999),
              ILS search on that subset.
    ils       all ambiguities; the search also runs on decorrelated ambiguities, which
              leaves the ILS solution unchanged (Z-invariance) and bounds the effort.
    bootstrap same subset as 'lambda', sequential conditional rounding instead of the search.
    cascade   no decorrelation; the most precise ambiguities (marginal sigma) are rounded
              one by one while the product of their rounding success rates reaches
              ``min_success``; correlation between satellite pairs is ignored.

    ``success`` is the exact IB success rate of the decorrelated fixed set (ILS lower bound,
    Ps-LAMBDA eq. 32) for lambda/ils/bootstrap, and the IR lower bound (eq. 26) of the rounded
    set for cascade. ``success_upper`` is the ADOP-based ILS upper bound (eq. 33)."""
    a = np.asarray(a, dtype=float); q = np.asarray(covariance, dtype=float)
    n = len(a)
    if method not in INTEGER_METHODS:
        raise ValueError(f'integer method must be one of {", ".join(INTEGER_METHODS)}')
    if method == 'cascade':
        sigma = np.sqrt(np.diag(q))
        order = np.argsort(sigma, kind='stable')
        rounding = np.array([math.erf(1/(2*math.sqrt(2)*v)) for v in sigma[order]])
        k = precise_prefix(rounding, min_success)
        combos = np.eye(n)[order[:k]]
        qz = combos@q@combos.T
        z_hat = combos@a
        integers, scores = runner_up(z_hat, qz, np.rint(z_hat))
        success = float(np.prod(rounding[:k]))
    else:
        z, _, d = decorrelate(q)
        if method == 'ils':
            k = n
        else:
            k = 0
            while k < n and bootstrap_success(d[n-k-1:]) >= min_success:
                k += 1
            k = max(k, 1)
        combos = z[:, n-k:].T
        qz = combos@q@combos.T
        z_hat = combos@a
        if method == 'bootstrap':
            integers, scores = runner_up(z_hat, qz, bootstrap(z_hat, qz))
        elif k == 1:
            # One-dimensional ILS is rounding; the runner-up is the next nearest integer.
            best = np.rint(z_hat); second = best+np.sign(z_hat-best+1e-15)
            integers = np.array([best, second])
            scores = np.array([((z_hat-c)**2/qz[0, 0]).item() for c in integers])
        else:
            integers, scores = search(z_hat, qz)
        success = None
        if method == 'ils':
            # Report integers of the original ambiguities: a = Z^-T z (Z unimodular).
            integers = np.rint(np.linalg.solve(combos, np.atleast_2d(integers).T)).T
            combos = np.eye(n)
    _, dz = ldl(qz)
    if success is None:
        success = bootstrap_success(dz)
    adop = float(np.exp(np.mean(np.log(dz))/2))
    return dict(combinations=combos, integers=integers[0], scores=scores, dimension=k,
                ratio=float(scores[1]/max(scores[0], 1e-12)), success=success,
                success_upper=adop_upper_bound(adop, k), adop=adop, float_dimension=n, method=method)
