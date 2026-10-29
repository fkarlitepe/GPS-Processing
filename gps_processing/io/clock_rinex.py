"""RINEX clock (3.00/3.04) reader: GPS and Galileo satellite clocks ('AS Gnn' / 'AS Enn' records).

High-rate clocks (30 s, 5 s) replace the 15 min SP3 clocks: a linear interpolation of 15 min
samples leaves decimetre-level clock errors, too large for narrow-lane ambiguity resolution.
"""
from datetime import datetime
import numpy as np
import pandas as pd
from gps_processing.io.ionex import read_text
from gps_processing import gnss


def read_clock_rinex(path, reference):
    """DataFrame(sat, epok, time): ``epok`` in seconds from ``reference`` (observation-day
    midnight), ``time`` in microseconds like the SP3 clock column."""
    lines = read_text(path)
    if not lines or 'C' not in lines[0][20:40] or 'RINEX VERSION' not in lines[0]:
        raise ValueError(f'{path}: RINEX clock header not found.')
    agency = ''
    rows = []
    for line in lines:
        if line[60:].startswith('ANALYSIS CENTER'):
            agency = line[:3].strip()
        elif line.startswith('AS ') and line[3:4] in gnss.SYSTEMS:
            fields = line[8:].split()
            y, mo, d, h, mi = (int(v) for v in fields[:5])
            seconds = float(fields[5])
            epoch = (datetime(y, mo, d, h, mi)-reference).total_seconds()+seconds
            rows.append((gnss.sat_id(line[3], int(line[4:6])), epoch, float(fields[7])*1e6))
    if not rows:
        raise ValueError(f'{path}: no GPS/Galileo satellite clock (AS G.. / AS E..) records.')
    frame = pd.DataFrame(rows, columns=['sat', 'epok', 'time']).drop_duplicates(['sat', 'epok'])
    frame.attrs['agency'] = agency
    return frame.sort_values(['sat', 'epok']).reset_index(drop=True)


class Clocks:
    """Linear interpolation inside a sampling interval only; gaps longer than ``max_gap`` raise."""

    def __init__(self, frame, max_gap=120.0):
        self.groups = {int(s): (f.epok.to_numpy(), f.time.to_numpy()) for s, f in frame.groupby('sat')}
        self.max_gap = max_gap

    def __call__(self, sat, epoch):
        if sat not in self.groups:
            raise ValueError(f'{gnss.label(sat)}: no satellite clock in the clock file.')
        t, value = self.groups[sat]
        k = np.searchsorted(t, epoch)
        if k < len(t) and abs(t[k]-epoch) < 1e-6:
            return value[k]
        if k == 0 or k == len(t) or t[k]-t[k-1] > self.max_gap:
            raise ValueError(f'{gnss.label(sat)}: epoch {epoch:.1f} s is outside or in a gap of the clock file.')
        w = (epoch-t[k-1])/(t[k]-t[k-1])
        return (1-w)*value[k-1]+w*value[k]

    def values(self, sat, epochs):
        """Vectorized ``__call__``: NaN where an epoch is outside the file or inside a gap."""
        epochs = np.asarray(epochs, float)
        out = np.full(len(epochs), np.nan)
        if sat not in self.groups or not len(epochs):
            return out
        t, value = self.groups[sat]
        k = np.searchsorted(t, epochs)
        exact = (k < len(t)) & (np.abs(t[np.minimum(k, len(t)-1)]-epochs) < 1e-6)
        out[exact] = value[k[exact]]
        inner = ~exact & (k > 0) & (k < len(t))
        lo, hi = k[inner]-1, k[inner]
        good = t[hi]-t[lo] <= self.max_gap
        w = (epochs[inner]-t[lo])/(t[hi]-t[lo])
        out[np.flatnonzero(inner)[good]] = ((1-w)*value[lo]+w*value[hi])[good]
        return out

    def covers(self, sat, epoch):
        try:
            self(sat, epoch)
            return True
        except ValueError:
            return False
