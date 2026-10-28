"""Receiver millisecond clock-jump detection and repair (pre-processing, all techniques).

Receivers that steer their clock in 1 ms steps shift either the code or the phase, not both.
Pc - Lc then jumps by k*c*1 ms on every satellite at once and the MW combination jumps by the
same amount, so the slip detector would restart every arc. Banville et al. (2008) show the
same common-mode jump in the fractional ambiguities after a clock slew. The jump is common
to all satellites and an exact multiple of 1 ms, so it is removed from the phase (Lc, A4)
instead of resetting the ambiguities; code and its time tag stay untouched.
"""
import numpy as np
from gps_processing.constants import C

MILLISECOND_M = C*1e-3


def repair_clock_jumps(combined, tolerance_m=30.0, agreement=0.8):
    """Return (repaired copy, [{'epok', 'jump_ms'}]); a jump needs >=2 continuing satellites,
    a median Δ(Pc-Lc) of at least 0.5 ms and ``agreement`` of them within ``tolerance_m``."""
    combined = combined.reset_index(drop=True)
    difference = (combined.Pc-combined.Lc).to_numpy()
    reset = combined.epoch_reset.to_numpy(bool)
    epochs = np.sort(combined.epok.unique())
    by_epoch = combined.reset_index(drop=True).groupby('epok').indices
    sats = combined.sat.to_numpy()
    shift, jumps = np.zeros(len(combined)), []
    cumulative = 0.0
    for previous, epoch in zip(epochs[:-1], epochs[1:]):
        now = by_epoch[epoch]
        before = dict(zip(sats[by_epoch[previous]], difference[by_epoch[previous]]-shift[by_epoch[previous]]))
        deltas = np.array([difference[i]-cumulative-before[sats[i]] for i in now if sats[i] in before and not reset[i]])
        if len(deltas) >= 2:
            k = np.rint(np.median(deltas)/MILLISECOND_M)
            if k != 0 and np.mean(np.abs(deltas-k*MILLISECOND_M) < tolerance_m) >= agreement:
                cumulative += k*MILLISECOND_M
                jumps.append(dict(epok=float(epoch), jump_ms=int(k)))
        shift[now] = cumulative
    # Code moved by +J relative to phase: add J to phase (Lc and the wide-lane phase in A4).
    combined['Lc'] += shift
    combined['A4'] += shift
    return combined, jumps
