"""Code and carrier combinations; all carrier inputs and outputs are metres."""
from gps_processing.constants import F1, F2

def ionosphere_free(p1, p2):
    return (F1**2 * p1 - F2**2 * p2) / (F1**2 - F2**2)

def melbourne_wubbena(p1, p2, l1, l2):
    return (F1*l1 - F2*l2)/(F1-F2) - (F1*p1 + F2*p2)/(F1+F2)
