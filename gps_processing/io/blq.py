"""Ocean tide loading coefficients, BLQ format (Onsala Space Observatory loading service).

One block per station: a name line, then six lines of 11 values for the constituents
M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa: amplitudes (m) up / west / south, then the Greenwich phase
lags (deg) up / west / south. Lines starting with '$$' are comments.
"""
from pathlib import Path
import numpy as np
from gps_processing.io.ionex import read_text

CONSTITUENTS = ('M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm', 'Ssa')


def read_blq(path):
    """{station name (upper case): (6, 11) array} with rows amp up/west/south, phase up/west/south."""
    stations, name, values = {}, None, []
    for line in read_text(path):
        if line.startswith('$$') or not line.strip():
            continue
        tokens = line.split()
        try:
            numbers = [float(t) for t in tokens]
        except ValueError:
            name, values = line.strip().upper(), []
            continue
        if name is None or len(numbers) != 11:
            continue
        values.append(numbers)
        if len(values) == 6:
            stations[name] = np.array(values)
            name, values = None, []
    if not stations:
        raise ValueError(f'{Path(path).name}: no BLQ ocean-loading block found.')
    return stations


def select_station(stations, marker, path=''):
    """Block of the RINEX marker: exact name, then the first four characters (IGS 4-char id)."""
    marker = (marker or '').strip().upper()
    for key in (marker, marker[:4]):
        if key and key in stations:
            return stations[key]
    for name, block in stations.items():
        if marker[:4] and name[:4] == marker[:4]:
            return block
    raise ValueError(f'{Path(str(path)).name or "BLQ"}: no ocean-loading coefficients for station {marker or "?"} '
                     f'(stations in file: {", ".join(sorted(stations)[:10])}{"…" if len(stations) > 10 else ""}).')
