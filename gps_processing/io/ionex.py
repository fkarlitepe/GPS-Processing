"""IONEX 1.0/1.1 global ionosphere maps (Schaer, Gurtner & Feltens 1998).

Only TEC maps are read (RMS and height maps are skipped). Values are returned in
TECU; missing cells (9999) become NaN.
"""
from dataclasses import dataclass
from datetime import datetime
import gzip
from pathlib import Path
import numpy as np


@dataclass
class IonexMaps:
    epochs: np.ndarray       # seconds relative to ``reference`` midnight
    latitudes: np.ndarray    # degrees, as listed in the file (usually 87.5 .. -87.5)
    longitudes: np.ndarray   # degrees (usually -180 .. 180)
    tec: np.ndarray          # TECU, shape (maps, latitudes, longitudes)
    height_km: float         # single-layer shell height (HGT1)
    radius_km: float         # base radius
    mapping: str             # MAPPING FUNCTION record (COSZ, QFAC, NONE)


def read_text(path):
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b'\x1f\x9d':
        raise ValueError(f'{path.name}: Unix-compressed (.Z) files are not supported; decompress it or use the .gz version.')
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return raw.decode('ascii', errors='replace').splitlines()


def epoch_seconds(fields, reference):
    y, mo, d, h, mi, s = (int(float(v)) for v in fields[:6])
    return (datetime(y, mo, d, h, mi, s)-reference).total_seconds()


def read_ionex(path, reference):
    """``reference`` is the midnight of the observation day (``Observations.date``)."""
    lines = read_text(path)
    if not lines or 'IONEX VERSION / TYPE' not in lines[0]:
        raise ValueError('IONEX file expected (first record must be "IONEX VERSION / TYPE").')
    header = dict(exponent=-1, height=450.0, radius=6371.0, mapping='COSZ')
    end = None
    for i, line in enumerate(lines):
        label = line[60:].strip()
        if label == 'EXPONENT':
            header['exponent'] = int(line[:6])
        elif label == 'HGT1 / HGT2 / DHGT':
            header['height'] = float(line[2:8])
        elif label == 'BASE RADIUS':
            header['radius'] = float(line[:8])
        elif label == 'MAPPING FUNCTION':
            header['mapping'] = line[:6].strip() or 'COSZ'
        elif label == 'LAT1 / LAT2 / DLAT':
            lat1, lat2, dlat = (float(line[k:k+6]) for k in (2, 8, 14))
        elif label == 'LON1 / LON2 / DLON':
            lon1, lon2, dlon = (float(line[k:k+6]) for k in (2, 8, 14))
        elif label == 'END OF HEADER':
            end = i+1
            break
    if end is None:
        raise ValueError('IONEX header is incomplete (END OF HEADER missing).')
    latitudes = np.arange(lat1, lat2+dlat/2, dlat)
    longitudes = np.arange(lon1, lon2+dlon/2, dlon)
    epochs, maps = [], []
    i, exponent = end, header['exponent']
    while i < len(lines):
        label = lines[i][60:].strip()
        if label == 'START OF TEC MAP':
            grid = np.full((len(latitudes), len(longitudes)), np.nan)
            i += 1
            while i < len(lines) and lines[i][60:].strip() != 'END OF TEC MAP':
                line, label = lines[i], lines[i][60:].strip()
                if label == 'EPOCH OF CURRENT MAP':
                    epochs.append(epoch_seconds(line[:36].split(), reference))
                elif label == 'EXPONENT':
                    exponent = int(line[:6])
                elif label == 'LAT/LON1/LON2/DLON/H':
                    lat = float(line[2:8])
                    row = int(round((lat-lat1)/dlat))
                    values = []
                    while len(values) < len(longitudes):
                        i += 1
                        values.extend(int(lines[i][k:k+5]) for k in range(0, len(lines[i].rstrip()), 5))
                    data = np.array(values[:len(longitudes)], dtype=float)
                    data[data == 9999] = np.nan
                    grid[row] = data*10.0**exponent
                i += 1
            maps.append(grid)
        elif label in ('START OF RMS MAP', 'START OF HEIGHT MAP'):
            stop = label.replace('START', 'END')
            while i < len(lines) and lines[i][60:].strip() != stop:
                i += 1
        i += 1
    if not maps or len(epochs) != len(maps):
        raise ValueError('IONEX file contains no complete TEC maps.')
    order = np.argsort(epochs)
    return IonexMaps(np.asarray(epochs)[order], latitudes, longitudes, np.asarray(maps)[order],
                     header['height'], header['radius'], header['mapping'])
