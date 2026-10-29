"""Bias-SINEX 1.00 reader (DSB / OSB, satellite entries) and code-bias alignment to C1W/C2W.

Banville et al. (2008, eq. 12): code biases propagate into the MW wide lane and, through the
IF satellite clocks, into the IF ambiguities, so the user code must be the one the clock and
bias products were made with (IGS: C1W/C2W). Bias-SINEX convention: observed = true + bias,
hence C1C - DSB(C1C, C1W) = C1C - (OSB(C1C) - OSB(C1W)) is the C1W-equivalent code.
Galileo codes are aligned the same way to the MGEX clock datum C1C/C5Q (RINEX 2 files only give
C1X/C5X, whose DSB to C1C/C5Q is in the multi-GNSS Bias-SINEX files).
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from gps_processing.constants import C
from gps_processing import gnss
from gps_processing.io.ionex import read_text

TARGET = {'G': {'1': 'C1W', '2': 'C2W'}, 'E': {'1': 'C1C', '5': 'C5Q'}}


@dataclass
class Bias:
    kind: str          # 'DSB' or 'OSB'
    prn: int           # integer satellite id (GPS n, Galileo 100+n)
    obs1: str
    obs2: str          # empty for OSB
    start: float       # seconds from the observation-day midnight
    end: float
    value_m: float


def sinex_time(text, reference):
    year, doy, seconds = (int(v) for v in text.split(':'))
    if year == doy == seconds == 0:
        return np.inf                     # open interval
    year += 2000 if year < 100 else 0
    return (datetime(year, 1, 1)+timedelta(days=doy-1, seconds=seconds)-reference).total_seconds()


def read_bias_sinex(path, reference):
    """Satellite GPS/Galileo DSB/OSB code biases in metres; ``reference`` is the observation-day midnight."""
    lines, inside, biases = read_text(path), False, []
    if not lines or not lines[0].startswith('%=BIA'):
        raise ValueError(f'{path}: Bias-SINEX header (%=BIA) not found.')
    for line in lines:
        if line.startswith('+BIAS/SOLUTION'):
            inside = True
        elif line.startswith('-BIAS/SOLUTION'):
            inside = False
        elif inside and line[:1] == ' ' and line[1:4] in ('DSB', 'OSB'):
            prn, station = line[11:14].strip(), line[15:24].strip()
            if station or prn[:1] not in gnss.SYSTEMS:
                continue                          # receiver biases and other systems are not used
            tokens = line[25:].split()
            obs1, tokens = tokens[0], tokens[1:]
            obs2 = '' if ':' in tokens[0] else tokens.pop(0)
            if not obs1.startswith('C'):
                continue                          # phase biases: see align_code_biases
            start, end, unit, value = tokens[:4]
            scale = {'ns': C*1e-9, 'm': 1.0}.get(unit)
            if scale is None:
                raise ValueError(f'{path}: unsupported bias unit {unit!r}.')
            biases.append(Bias(line[1:4], gnss.parse(prn), obs1, obs2, sinex_time(start, reference),
                               sinex_time(end, reference), float(value)*scale))
    if not biases:
        raise ValueError(f'{path}: no satellite GPS/Galileo code biases in +BIAS/SOLUTION.')
    return biases


class CodeBiases:
    def __init__(self, biases):
        self.biases = biases

    def _value(self, prn, obs1, obs2, epoch):
        def find(kind, a, b):
            for item in self.biases:
                if item.kind == kind and item.prn == prn and item.obs1 == a and item.obs2 == b and item.start <= epoch < item.end:
                    return item.value_m
            return None
        direct = find('DSB', obs1, obs2)
        if direct is not None:
            return direct
        reverse = find('DSB', obs2, obs1)
        if reverse is not None:
            return -reverse
        one, two = find('OSB', obs1, ''), find('OSB', obs2, '')
        return one-two if one is not None and two is not None else None

    def dsb(self, prn, code, epoch):
        """DSB(code, clock-datum code of its band) in metres (0 for the datum code itself), None
        when the file lacks it; ``prn`` is the integer satellite id."""
        target = TARGET[gnss.system_of(prn)][code[1]]
        return 0.0 if code == target else self._value(prn, code, target, epoch)


def align_code_biases(combined, biases):
    """Shift codes to the clock-datum codes (GPS C1W/C2W, Galileo C1C/C5Q); Pc, A4 and travel_code
    are linear in the codes.

    Phase is left unchanged: under the IGS convention the satellite phase bias of one frequency
    does not depend on the tracking mode, and receiver-side offsets (RINEX 3 phase shifts) are
    common to all satellites and absorbed by the receiver clock / ambiguity datum. Rows whose
    bias is missing keep their original signal label, so the AR product check still rejects them.
    """
    combined = combined.copy()
    combined['code_bias_aligned'] = False
    codes = combined.signals.str.split(expand=True)
    for (sat, c1, c2), index in combined.groupby([combined.sat, codes[0], codes[2]]).groups.items():
        target = TARGET[gnss.system_of(sat)]
        one, two = target[c1[1]], target.get(c2[1])
        if two is None or (c1 == one and c2 == two):
            continue
        rows = combined.loc[index]
        d1 = np.array([np.nan if (v := biases.dsb(int(sat), c1, e)) is None else v for e in rows.epok])
        d2 = np.array([np.nan if (v := biases.dsb(int(sat), c2, e)) is None else v for e in rows.epok])
        ok = np.isfinite(d1) & np.isfinite(d2)
        index = rows.index[ok]
        d1, d2 = d1[ok], d2[ok]
        k = gnss.constants(gnss.system_of(sat))
        f1, f2 = k['f1'], k['f2']
        combined.loc[index, 'Pc'] -= (f1**2*d1-f2**2*d2)/(f1**2-f2**2)
        combined.loc[index, 'A4'] += (f1*d1+f2*d2)/(f1+f2)
        if 'travel_code' in combined:
            combined.loc[index, 'travel_code'] -= d1
        parts = codes.loc[index]
        combined.loc[index, 'signals'] = (one+' '+parts[1]+' '+two+' '+parts[3]).to_numpy()
        combined.loc[index, 'code_bias_aligned'] = True
    return combined
