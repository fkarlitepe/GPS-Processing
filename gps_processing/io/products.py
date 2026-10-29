"""One entry point for PPP-AR products.

Every product ends up as the same three per-observation corrections added by the AR solver
(metres, c*dt_sat included): ``code_m`` (IF code), ``phase_m`` (IF phase) and ``mw_m`` (MW):

* Bias-SINEX OSB (*.BIA, *.BSX): code and phase OSB per tracked signal (integer-clock products);
* GENFCB FCB (*.fcb): L1/L2 (E1/E5a) phase FCBs per interval;
* TOGU JSON (togu-ar-1): the corrections themselves per epoch and satellite.
"""
import re
from pathlib import Path


def is_osb(path):
    """Bias-SINEX (*.BIA, *.BSX, gz) AR products carry code and phase OSBs; others are FCB or JSON."""
    return bool(re.search(r'\.(BIA|BSX)(\.GZ)?$', str(path).upper()))


def kind(path):
    if is_osb(path):
        return 'osb'
    return 'fcb' if str(path).lower().endswith('.fcb') else 'json'


def load(path, method, date, clock_file=''):
    """Reader object with ``meta``, ``frame`` and ``apply(observations)`` for ``path``."""
    found = kind(path)
    if found == 'osb':
        from gps_processing.io.osb_products import OSBProducts
        return OSBProducts(path, method, date, clock_file)
    if found == 'fcb':
        from gps_processing.io.fcb_products import FCBProducts
        return FCBProducts(path, method, date)
    from gps_processing.io.ar_products import ARProducts
    return ARProducts(path, method, date)
