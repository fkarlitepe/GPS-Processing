"""Code and carrier combinations; all carrier inputs and outputs are metres.

``f1``/``f2`` default to GPS L1/L2 and may be per-row arrays (Galileo E1/E5a, see gnss.frequencies).
"""
from gps_processing.constants import F1, F2

def ionosphere_free(p1, p2, f1=F1, f2=F2):
    return (f1**2 * p1 - f2**2 * p2) / (f1**2 - f2**2)

def melbourne_wubbena(p1, p2, l1, l2, f1=F1, f2=F2):
    return (f1*l1 - f2*l2)/(f1-f2) - (f1*p1 + f2*p2)/(f1+f2)
