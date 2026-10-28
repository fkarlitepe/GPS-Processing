"""SINEX_TRO troposphere solutions (IGS station ZTD products, v0.01 and v2.00).

Reads the +TROP/SOLUTION block: total zenith delay (TROTOT) and, when present,
total north/east gradients (TGNTOT/TGETOT). Units follow the "TROPO PARAMETER
UNITS" record (IGS default: millimetres, i.e. factor 1e+03).
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from gps_processing.io.ionex import read_text


@dataclass
class TroposphereSeries:
    site: str
    epochs: np.ndarray            # seconds relative to the observation-day midnight
    ztd: np.ndarray               # m
    ztd_sigma: np.ndarray         # m (NaN when not given)
    gradient_north: np.ndarray    # m (zeros when not given)
    gradient_east: np.ndarray     # m
    xyz: np.ndarray = None        # station coordinate from +TROP/STA_COORDINATES


def sinex_epoch(text, reference):
    year, doy, seconds = text.split(':')
    year = int(year)
    if year < 100:
        year += 2000 if year < 80 else 1900
    return (datetime(year, 1, 1)+timedelta(days=int(doy)-1, seconds=int(seconds))-reference).total_seconds()


def blocks(lines):
    current, out = None, {}
    for line in lines:
        if line.startswith('+'):
            current = line[1:].split()[0]
            out[current] = []
        elif line.startswith('-'):
            current = None
        elif current is not None:
            out[current].append(line)
    return out


def read_sinex_tro(path, reference):
    lines = read_text(path)
    if not lines or not lines[0].startswith('%=TRO'):
        raise ValueError('SINEX_TRO file expected (first record must start with "%=TRO").')
    sections = blocks(lines)
    if 'TROP/SOLUTION' not in sections:
        raise ValueError('SINEX_TRO file has no +TROP/SOLUTION block.')
    names, factors = None, {}
    for line in sections.get('TROP/DESCRIPTION', []):
        text = line.strip()
        if text.startswith('TROPO PARAMETER NAMES'):
            names = text[len('TROPO PARAMETER NAMES'):].split()
        elif text.startswith('TROPO PARAMETER UNITS'):
            units = [float(v) for v in text[len('TROPO PARAMETER UNITS'):].split()]
            factors = dict(zip(names or [], units))
    rows = [line for line in sections['TROP/SOLUTION'] if line.strip() and not line.startswith('*')]
    titles = [line for line in sections['TROP/SOLUTION'] if line.startswith('*SITE')]
    if titles:
        names = titles[0].split()[2:]
    if not names or 'TROTOT' not in names:
        raise ValueError('SINEX_TRO solution has no TROTOT (total zenith delay) column.')
    # Each STDDEV column belongs to the parameter before it.
    names = [f'{names[k-1]}_STDDEV' if name == 'STDDEV' and k else name for k, name in enumerate(names)]
    coordinates = {}
    for line in sections.get('TROP/STA_COORDINATES', []):
        if line.startswith('*') or not line.strip():
            continue
        parts = line.split()
        coordinates[parts[0].upper()] = np.array([float(v) for v in parts[4:7]])
    series = {}
    for line in rows:
        parts = line.split()
        record = dict(zip(names, (float(v) for v in parts[2:])))
        series.setdefault(parts[0].upper(), []).append((sinex_epoch(parts[1], reference), record))
    out = {}
    for site, records in series.items():
        records.sort(key=lambda item: item[0])
        def column(name, default):
            scale = 1/factors.get(name.split('_')[0], 1e3)
            return np.array([r.get(name, np.nan) for _, r in records])*scale if name in records[0][1] \
                else np.full(len(records), default)
        out[site] = TroposphereSeries(site, np.array([e for e, _ in records]), column('TROTOT', np.nan),
                                      column('TROTOT_STDDEV', np.nan), column('TGNTOT', 0.0),
                                      column('TGETOT', 0.0), coordinates.get(site))
    return out


def select_site(series, marker, xyz, max_distance=20e3):
    """Pick the RINEX marker's series, otherwise the nearest station within ``max_distance``."""
    marker = (marker or '')[:4].upper()
    for site, item in series.items():
        if site[:4] == marker:
            return item
    located = [(np.linalg.norm(item.xyz-xyz), item) for item in series.values() if item.xyz is not None]
    if located:
        distance, item = min(located, key=lambda pair: pair[0])
        if distance <= max_distance:
            return item
    if len(series) == 1 and not located:
        return next(iter(series.values()))
    raise ValueError(f'SINEX_TRO file has no solution for station {marker or "?"} '
                     f'(stations: {", ".join(sorted(series)[:10])}).')
