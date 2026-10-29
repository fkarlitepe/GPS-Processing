"""Observable-specific signal biases (Bias-SINEX OSB, code and phase) as PPP-AR products.

Integer-clock products (CODE, GFZ, WHU, CNES via OSB; Laurichesse et al. 2009, Schaer et al.
2021) give, with their own orbits and clocks, satellite code and phase OSBs such that
observation - OSB has integer undifferenced ambiguities. They are converted here to the three
corrections the AR solver adds to the observations (metres): IF code, IF phase and MW. The
phase OSB of one frequency is taken from any tracking mode of that frequency (IGS convention:
L1C = L1W, L2W = L2L for the satellite phase bias).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.constants import C, F1, F2, F5
from gps_processing import gnss
from gps_processing.io.bias_sinex import sinex_time
from gps_processing.io.ionex import read_text

LAMBDA_W = C/(F1-F2)


def read_osb(path, reference):
    """Satellite OSBs on GPS L1/L2 and Galileo E1/E5a: DataFrame(sat, obs, start, end, m)."""
    bands = {system: ''.join(info['bands']) for system, info in gnss.SYSTEMS.items()}
    lines = read_text(path)
    if not lines or not lines[0].startswith('%=BIA'):
        raise ValueError(f'{Path(path).name}: Bias-SINEX header (%=BIA) not found.')
    rows, inside = [], False
    for line in lines:
        if line.startswith('+BIAS/SOLUTION'):
            inside = True
        elif line.startswith('-BIAS/SOLUTION'):
            break
        elif (inside and line.startswith(' OSB') and line[11:12] in bands and not line[15:24].strip()
              and line[25:26] in 'CL' and line[26:27] in bands[line[11:12]]):
            tokens = line[25:].split()
            unit = tokens[3]
            if unit != 'ns' and not (unit == 'cyc' and tokens[0][0] == 'L'):
                raise ValueError(f'{Path(path).name}: unsupported OSB unit {unit!r}.')
            value = float(tokens[4])
            if unit == 'ns':
                value *= C*1e-9
            else:
                value *= C/{'1': F1, '2': F2, '5': F5}[tokens[0][1]]
            rows.append((gnss.sat_id(line[11], int(line[12:14])), tokens[0], sinex_time(tokens[1], reference),
                         sinex_time(tokens[2], reference), value))
    frame = pd.DataFrame(rows, columns=['sat', 'obs', 'start', 'end', 'm'])
    if frame.empty or not frame.obs.str.startswith('L').any():
        raise ValueError(f'{Path(path).name}: no GPS/Galileo phase OSB; integer-clock AR needs code and phase OSBs.')
    frame.attrs['agency'] = lines[0][6:15].split()[-1] if len(lines[0]) > 15 else ''
    return frame.sort_values(['sat', 'obs', 'start']).reset_index(drop=True)


def satellite_clock_m(observations):
    """c*dt_sat (m) from the interpolated SP3/CLK clock column (microseconds)."""
    if 'dt_sat' not in observations:
        raise ValueError('Satellite clocks (dt_sat) must be interpolated before AR products are applied.')
    return C*1e-6*observations.dt_sat.to_numpy(float)


class OSBProducts:
    """Same interface as io.ar_products.ARProducts: ``meta`` and ``apply(observations)``."""

    def __init__(self, path, method, date, clock_file=''):
        self.frame = read_osb(path, date)
        self.groups = {key: (g.start.to_numpy(), g.end.to_numpy(), g.m.to_numpy())
                       for key, g in self.frame.groupby(['sat', 'obs'])}
        name = Path(path).name
        if clock_file and len(name) > 9 and len(Path(clock_file).name) > 9 and name[:3] != Path(clock_file).name[:3]:
            raise ValueError(f'OSB product {name} and clock file {Path(clock_file).name} come from different '
                             'analysis centres; phase biases are only valid with their own clocks.')
        self.meta = dict(schema='bias-sinex-osb', method=method, date=date.strftime('%Y-%m-%d'), source=name,
                         clock_product=Path(clock_file).name if clock_file else '',
                         phase_observables=sorted(self.frame.obs[self.frame.obs.str.startswith('L')].unique()))

    def value(self, sat, obs, epochs):
        """OSB (m) at ``epochs``; NaN outside every validity interval."""
        candidates = [obs] if obs[0] == 'C' else [obs]+sorted(o for s, o in self.groups if s == sat and o[:2] == obs[:2] and o != obs)
        out = np.full(len(epochs), np.nan)
        for name in candidates:
            if (sat, name) not in self.groups:
                continue
            start, end, value = self.groups[sat, name]
            k = np.searchsorted(start, epochs, side='right')-1
            ok = (k >= 0) & np.isnan(out)
            k = np.clip(k, 0, None)
            ok &= epochs < end[k]
            out[ok] = value[k[ok]]
        return out

    def apply(self, observations):
        result = observations.copy()
        if 'signals' not in result:
            raise ValueError('Observations have no signal labels; OSBs cannot be matched.')
        biases = np.full((len(result), 4), np.nan)
        signals = result.signals.str.split(expand=True)
        for (sat, *names), index in result.groupby([result.sat, signals[0], signals[1], signals[2], signals[3]]).groups.items():
            epochs = result.loc[index, 'epok'].to_numpy(float)
            rows = result.index.get_indexer(index)
            for j, name in enumerate(names):
                biases[rows, j] = self.value(int(sat), name, epochs)
        c1, l1, c2, l2 = biases.T
        f1, f2 = gnss.frequencies(result.sat.to_numpy())          # GPS L1/L2, Galileo E1/E5a
        # The AR solver models no satellite clock: like the JSON products, code_m/phase_m carry c*dt_sat.
        clock = satellite_clock_m(result)
        result['code_m'] = clock-(f1**2*c1-f2**2*c2)/(f1**2-f2**2)
        result['phase_m'] = clock-(f1**2*l1-f2**2*l2)/(f1**2-f2**2)
        result['mw_m'] = -((f1*l1-f2*l2)/(f1-f2)-(f1*c1+f2*c2)/(f1+f2))
        self.meta['systems'] = sorted({gnss.system_of(s) for s in result.sat[np.isfinite(biases).all(axis=1)]})
        complete = np.isfinite(biases).all(axis=1)
        self.meta['dropped_rows'] = int((~complete).sum())
        if not complete.any():
            raise ValueError('No observation has complete code and phase OSBs (check day and signals).')
        return result[complete].reset_index(drop=True)
