"""Uncombined fractional-cycle-bias (FCB / UPD) products, GENFCB text format, as PPP-AR products.

Each record gives the L1 and L2 satellite phase biases (cycles) for an interval:
    2023/11/23 00:00:00 2023/11/23 00:05:00  G05  -1.85653 0.08586  -2.22089 0.10825 ...
Phase in cycles contains N_i + b_i, so the corrections added to the observations are
    phase_m = -(IF combination of lambda_i * b_i),  mw_m = -lambda_W (b_1 - b_2),  code_m = 0
(WL check on real data: MW/lambda_W - (b_1 - b_2) is integer up to the receiver bias).
FCBs refer to the orbit/clock product used to estimate them; the narrow-lane part is only valid
with that product, which the file does not name.
"""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.constants import C, F1, F2
from gps_processing.io.ionex import read_text

IF1, IF2 = F1**2/(F1**2-F2**2), -F2**2/(F1**2-F2**2)
LAMBDA_W = C/(F1-F2)


def read_fcb(path, reference):
    lines = read_text(path)
    columns = None
    rows = []
    for line in lines:
        if line.startswith('#s'):
            columns = line.split()[2:]
        elif line[:1].isdigit():
            p = line.split()
            if len(p) < 7 or not p[4].startswith('G'):
                continue
            start = (datetime.strptime(p[0]+' '+p[1], '%Y/%m/%d %H:%M:%S')-reference).total_seconds()
            end = (datetime.strptime(p[2]+' '+p[3], '%Y/%m/%d %H:%M:%S')-reference).total_seconds()
            rows.append((int(p[4][1:]), start, end, float(p[5]), float(p[7])))
    if columns is None or columns[:2] != ['L1', 'L2'] or not rows:
        raise ValueError(f'{Path(path).name}: not a GENFCB L1/L2 FCB file.')
    return pd.DataFrame(rows, columns=['sat', 'start', 'end', 'b1', 'b2']).sort_values(['sat', 'start']).reset_index(drop=True)


class FCBProducts:
    """Same interface as io.ar_products.ARProducts."""

    def __init__(self, path, method, date):
        self.frame = read_fcb(path, date)
        self.groups = {int(s): g for s, g in self.frame.groupby('sat')}
        self.meta = dict(schema='genfcb', method=method, date=date.strftime('%Y-%m-%d'), source=Path(path).name,
                         note='FCB narrow-lane part is valid only with the orbit/clock product used to estimate it')

    def apply(self, observations):
        result = observations.copy()
        b1 = np.full(len(result), np.nan); b2 = b1.copy()
        for sat, index in result.groupby('sat').groups.items():
            if int(sat) not in self.groups:
                continue
            g = self.groups[int(sat)]
            epochs = result.loc[index, 'epok'].to_numpy(float)
            k = np.clip(np.searchsorted(g.start.to_numpy(), epochs, side='right')-1, 0, None)
            ok = (epochs >= g.start.to_numpy()[k]) & (epochs < g.end.to_numpy()[k])
            rows = result.index.get_indexer(index)
            b1[rows[ok]], b2[rows[ok]] = g.b1.to_numpy()[k[ok]], g.b2.to_numpy()[k[ok]]
        from gps_processing.io.osb_products import satellite_clock_m
        clock = satellite_clock_m(result)                  # c*dt_sat, as in the JSON product contract
        result['code_m'] = clock
        result['phase_m'] = clock-(IF1*C/F1*b1+IF2*C/F2*b2)
        result['mw_m'] = -LAMBDA_W*(b1-b2)
        complete = np.isfinite(b1) & np.isfinite(b2)
        self.meta['dropped_rows'] = int((~complete).sum())
        if not complete.any():
            raise ValueError('No observation is covered by the FCB file.')
        return result[complete].reset_index(drop=True)
