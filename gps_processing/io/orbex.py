"""ORBEX attitude quaternions (ATT records), e.g. GBM0MGXRAP_*_ATT.OBX of the GFZ/CODE products.

Each ``## yyyy mm dd hh mm ss n`` epoch is followed by ``ATT Gnn 4 q0 q1 q2 q3`` records
(scalar first). R(q) maps ECEF vectors into the body frame, r_body = R(q) r_ecef, so the rows of
R are the body x, y, z axes in ECEF (checked on GBM: z towards the geocentre within 0.01 deg and
x within 0.1 deg of the nominal yaw outside eclipse). With the
analysis centre's own attitude the PCO and the phase wind-up follow the real yaw during
eclipse and noon turns, where the nominal yaw is wrong by up to 180 degrees.
"""
from datetime import datetime
from pathlib import Path
import numpy as np
from gps_processing.io.ionex import read_text
from gps_processing import gnss


def read_orbex(path, reference):
    """{sat: (epochs (s from ``reference``), quaternions (n, 4))} for GPS and Galileo."""
    lines = read_text(path)
    if not lines or not lines[0].startswith('%=ORBEX'):
        raise ValueError(f'{Path(path).name}: ORBEX header (%=ORBEX) not found.')
    frame = next((line.split()[1] for line in lines[:60] if line.split()[:1] == ['FRAME_TYPE']), 'ECEF')
    if frame != 'ECEF':
        raise ValueError(f'{Path(path).name}: attitude frame {frame} is not supported (ECEF only).')
    data, epoch = {}, None
    for line in lines:
        if line.startswith('##'):
            v = line[2:].split()
            epoch = (datetime(*(int(x) for x in v[:5]))-reference).total_seconds()+float(v[5])
        elif line.startswith(' ATT ') and line[5:6] in gnss.SYSTEMS and epoch is not None:
            v = line[5:].split()
            try:
                sat = gnss.parse(v[0])
            except ValueError:
                continue
            data.setdefault(sat, []).append((epoch, *map(float, v[2:6])))
    if not data:
        raise ValueError(f'{Path(path).name}: no GPS/Galileo ATT records.')
    out = {}
    for sat, rows in data.items():
        rows = np.array(sorted(rows))
        out[sat] = (rows[:, 0], rows[:, 1:])
    return out


def rotation(q):
    """(n, 3, 3) rotation matrices of unit quaternions (q0 scalar)."""
    q = q/np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)


def axes(attitude, sats, epochs, max_gap=600.0):
    """Body x, y, z axes (ECEF rows) per observation and a mask of rows the file covers.

    Quaternions are interpolated linearly (normalized, sign-aligned) between records at most
    ``max_gap`` seconds apart; other rows keep NaN and fall back to the nominal yaw."""
    n = len(sats)
    out = np.full((3, n, 3), np.nan)
    covered = np.zeros(n, bool)
    for sat in np.unique(sats):
        if int(sat) not in attitude:
            continue
        t, q = attitude[int(sat)]
        rows = np.flatnonzero(sats == sat)
        e = epochs[rows]
        k = np.clip(np.searchsorted(t, e)-1, 0, len(t)-2)
        ok = (e >= t[k]) & (e <= t[k+1]) & (t[k+1]-t[k] <= max_gap) if len(t) > 1 else np.zeros(len(e), bool)
        if not ok.any():
            continue
        k, rows, e = k[ok], rows[ok], e[ok]
        a, b = q[k], q[k+1]
        b = np.where((np.sum(a*b, axis=1) < 0)[:, None], -b, b)
        w = ((e-t[k])/(t[k+1]-t[k]))[:, None]
        r = rotation((1-w)*a+w*b)
        for axis in range(3):
            out[axis, rows] = r[:, axis, :]
        covered[rows] = True
    return out[0], out[1], out[2], covered
