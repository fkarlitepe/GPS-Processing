"""Integer phase clock model (Laurichesse et al. 2008), user side as in Shi & Gao (2014).

Wide lane: the MW combination corrected with the satellite wide-lane bias (WSB) still
contains the wide-lane receiver bias (WRB), common to all satellites. The WRB is the
satellite average of the fractional parts (eq. 46, taken as a circular mean so that
fractions near +-0.5 do not cancel) and the integer wide lane follows by rounding
(eq. 47). N1: with the WL known, IF code and phase observe position, troposphere, a
receiver code clock, a receiver phase clock and the N1 ambiguities directly (eqs. 21, 48);
the MW rows therefore do not enter the filter.
"""
import numpy as np
from .models import LAMBDA_W


def transform(y, predicted, design, noise, reference):
    """Keep the IF code and phase rows (Pc, Lc) of every satellite; drop MW (``reference`` unused)."""
    keep = np.sort(np.r_[np.arange(0, len(y), 3), np.arange(1, len(y), 3)])
    return y[keep], predicted[keep], design[keep], noise[np.ix_(keep, keep)]


class WideLaneResolver:
    """Arc-averaged MW per satellite, satellite-averaged WRB and rounded wide lanes."""

    def __init__(self, min_epochs=20, max_fraction=0.25, max_sigma=0.1, lambda_w=LAMBDA_W):
        """One resolver per satellite system: the receiver WL bias and wavelength are system specific."""
        self.min_epochs, self.max_fraction, self.max_sigma, self.lambda_w = min_epochs, max_fraction, max_sigma, lambda_w
        self.sum, self.square, self.count = {}, {}, {}
        self.receiver_bias = np.nan

    def reset(self, sat):
        for store in (self.sum, self.square, self.count):
            store.pop(sat, None)

    def add(self, sat, mw_m):
        cycles = mw_m/self.lambda_w
        self.sum[sat] = self.sum.get(sat, 0.0)+cycles
        self.square[sat] = self.square.get(sat, 0.0)+cycles**2
        self.count[sat] = self.count.get(sat, 0)+1

    def mean(self, sat):
        return self.sum[sat]/self.count[sat]

    def sigma(self, sat):
        """Standard deviation of the arc mean (cycles), from the scatter of the arc."""
        k = self.count[sat]
        if k < 2:
            return np.inf
        variance = max(self.square[sat]/k-self.mean(sat)**2, 0.0)*k/(k-1)
        return float(np.sqrt(variance/k))

    def resolve(self, satellites):
        """Integer WL (undifferenced, still including the integer part of the WRB) of every
        satellite whose arc passes the gates; WRB = circular mean of the fractional parts."""
        ready = [s for s in satellites if self.count.get(s, 0) >= self.min_epochs and self.sigma(s) < self.max_sigma]
        if len(ready) < 2:
            return {}
        means = np.array([self.mean(s) for s in ready])
        self.receiver_bias = float(np.angle(np.mean(np.exp(2j*np.pi*means)))/(2*np.pi))
        corrected = means-self.receiver_bias
        integers = np.rint(corrected)
        return {s: int(n) for s, n, c in zip(ready, integers, corrected) if abs(c-n) < self.max_fraction}
