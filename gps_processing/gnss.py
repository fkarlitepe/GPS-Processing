"""Satellite systems: identifiers, frequencies and derived constants (GPS L1/L2, Galileo E1/E5a, BDS B1I/B3I).

Satellites are integers throughout the code: GPS PRN n -> n (1..32), Galileo PRN n -> 100+n
(101..136), BDS PRN n -> 200+n (201..263; geostationary satellites are not used). ``slot`` gives a
dense 0-based index for filter state vectors (GPS 0..31, Galileo 32..67, BDS 68..130).
"""
import numpy as np
from gps_processing.constants import C, F1, F2, F5

F_B1I, F_B3I = 1561098000.0, 1268520000.0

SYSTEMS = {
    'G': dict(name='GPS', offset=0, count=32, bands=('1', '2'), f1=F1, f2=F2),
    'E': dict(name='Galileo', offset=100, count=36, bands=('1', '5'), f1=F1, f2=F5),
    'C': dict(name='BDS', offset=200, count=63, bands=('2', '6'), f1=F_B1I, f2=F_B3I),
}
BDS_GEO = {201, 202, 203, 204, 205, 259, 260, 261, 262, 263}
NAMES = {key: value['name'] for key, value in SYSTEMS.items()}
SLOTS = SYSTEMS['G']['count']+SYSTEMS['E']['count']        # 68 state slots for GPS + Galileo
SLOTS_ALL = sum(s['count'] for s in SYSTEMS.values())       # 131 with BDS
_START = {'G': 0, 'E': SYSTEMS['G']['count'], 'C': SLOTS}


def slots_for(systems):
    """Number of ambiguity slots of a filter that processes ``systems``."""
    return SLOTS_ALL if 'C' in systems else SLOTS


def sat_id(system, prn):
    if system not in SYSTEMS:
        raise ValueError(f'Unsupported satellite system {system!r}.')
    if not 1 <= int(prn) <= SYSTEMS[system]['count']:
        raise ValueError(f'{system}{int(prn):02d}: PRN out of range.')
    return SYSTEMS[system]['offset']+int(prn)


def system_of(sat):
    sat = int(sat)
    return 'C' if sat > 200 else 'E' if sat > 100 else 'G'


def prn_of(sat):
    return int(sat)-SYSTEMS[system_of(sat)]['offset']


def label(sat):
    return f'{system_of(sat)}{prn_of(sat):02d}'


def parse(text):
    """'G05' / 'E12' / 'C25' -> integer id."""
    return sat_id(text[0], int(text[1:3]))


def slot(sat):
    """Dense 0-based index (GPS 0..31, Galileo 32..67, BDS 68..130)."""
    return _START[system_of(sat)]+prn_of(sat)-1


def systems_of(sats):
    sats = np.atleast_1d(np.asarray(sats, int))
    return np.where(sats > 200, 'C', np.where(sats > 100, 'E', 'G'))


def frequencies(sats):
    """(f1, f2) arrays for integer satellite ids (scalar or array)."""
    sats = np.atleast_1d(np.asarray(sats, int))
    systems = systems_of(sats)
    f1 = np.select([systems == 'C', systems == 'E'], [F_B1I, SYSTEMS['E']['f1']], F1)
    f2 = np.select([systems == 'C', systems == 'E'], [F_B3I, SYSTEMS['E']['f2']], F2)
    return f1, f2


def constants(system):
    """Wavelengths and IF coefficients of one system (metres, dimensionless)."""
    f1, f2 = SYSTEMS[system]['f1'], SYSTEMS[system]['f2']
    return dict(f1=f1, f2=f2, lambda1=C/f1, lambda2=C/f2, lambda_w=C/(f1-f2), lambda_n=C/(f1+f2),
                beta=C*f2/(f1**2-f2**2), if1=f1**2/(f1**2-f2**2), if2=-f2**2/(f1**2-f2**2),
                if_factor=np.sqrt(f1**4+f2**4)/(f1**2-f2**2))


def per_row(sats, key):
    """Constant ``key`` for each satellite id."""
    values = {s: constants(s)[key] for s in SYSTEMS}
    systems = systems_of(sats)
    return np.select([systems == 'C', systems == 'E'], [values['C'], values['E']], values['G'])
