"""Satellite systems: identifiers, frequencies and derived constants (GPS L1/L2, Galileo E1/E5a).

Satellites are integers throughout the code: GPS PRN n -> n (1..32), Galileo PRN n -> 100+n
(101..136). ``slot`` gives a dense 0-based index for filter state vectors.
"""
import numpy as np
from gps_processing.constants import C, F1, F2, F5

SYSTEMS = {
    'G': dict(name='GPS', offset=0, count=32, bands=('1', '2'), f1=F1, f2=F2),
    'E': dict(name='Galileo', offset=100, count=36, bands=('1', '5'), f1=F1, f2=F5),
}
NAMES = {key: value['name'] for key, value in SYSTEMS.items()}
SLOTS = sum(s['count'] for s in SYSTEMS.values())          # 68 state slots
_START = {'G': 0, 'E': SYSTEMS['G']['count']}


def sat_id(system, prn):
    if system not in SYSTEMS:
        raise ValueError(f'Unsupported satellite system {system!r}.')
    if not 1 <= int(prn) <= SYSTEMS[system]['count']:
        raise ValueError(f'{system}{int(prn):02d}: PRN out of range.')
    return SYSTEMS[system]['offset']+int(prn)


def system_of(sat):
    return 'E' if int(sat) > 100 else 'G'


def prn_of(sat):
    return int(sat)-SYSTEMS[system_of(sat)]['offset']


def label(sat):
    return f'{system_of(sat)}{prn_of(sat):02d}'


def parse(text):
    """'G05' / 'E12' -> integer id."""
    return sat_id(text[0], int(text[1:3]))


def slot(sat):
    """Dense 0-based index (GPS 0..31, Galileo 32..67)."""
    return _START[system_of(sat)]+prn_of(sat)-1


def systems_of(sats):
    return np.array(['E' if s > 100 else 'G' for s in np.atleast_1d(sats)])


def frequencies(sats):
    """(f1, f2) arrays for integer satellite ids (scalar or array)."""
    sats = np.atleast_1d(np.asarray(sats, int))
    galileo = sats > 100
    return np.where(galileo, SYSTEMS['E']['f1'], F1), np.where(galileo, SYSTEMS['E']['f2'], F2)


def constants(system):
    """Wavelengths and IF coefficients of one system (metres, dimensionless)."""
    f1, f2 = SYSTEMS[system]['f1'], SYSTEMS[system]['f2']
    return dict(f1=f1, f2=f2, lambda1=C/f1, lambda2=C/f2, lambda_w=C/(f1-f2), lambda_n=C/(f1+f2),
                beta=C*f2/(f1**2-f2**2), if1=f1**2/(f1**2-f2**2), if2=-f2**2/(f1**2-f2**2),
                if_factor=np.sqrt(f1**4+f2**4)/(f1**2-f2**2))


def per_row(sats, key):
    """Constant ``key`` for each satellite id."""
    values = {s: constants(s)[key] for s in SYSTEMS}
    return np.where(np.atleast_1d(np.asarray(sats, int)) > 100, values['E'], values['G'])
