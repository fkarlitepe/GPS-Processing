"""Solution quality: local ENU errors, convergence time and method statistics."""
import numpy as np
import pandas as pd
from gps_processing.processing.troposphere import geodetic


def enu_rotation(xyz):
    lat, lon, _ = geodetic(xyz)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0], [-sl*co, -sl*so, cl], [cl*co, cl*so, sl]])


def add_enu(positions, reference):
    """Append east/north/up and horizontal/3D errors (m) relative to ``reference``."""
    reference = np.asarray(reference, dtype=float)
    delta = positions[['x', 'y', 'z']].to_numpy(dtype=float)-reference
    enu = delta@enu_rotation(reference).T
    frame = positions.copy()
    frame['east'], frame['north'], frame['up'] = enu.T
    frame['error_h'] = np.hypot(enu[:, 0], enu[:, 1])
    frame['error_3d'] = np.linalg.norm(enu, axis=1)
    return frame


def convergence_time(frame, horizontal=0.10, vertical=0.20):
    """Seconds from the first epoch until errors stay below the thresholds; NaN if never."""
    ok = (frame.error_h <= horizontal) & (frame.up.abs() <= vertical)
    bad = np.flatnonzero(~ok.to_numpy())
    if len(bad) == len(frame):
        return np.nan
    start = 0 if len(bad) == 0 else bad[-1]+1
    if start >= len(frame):
        return np.nan
    return float(frame.epok.iloc[start]-frame.epok.iloc[0])


def statistics(positions, reference, label='', reference_source='', horizontal=0.10, vertical=0.20):
    frame = add_enu(positions, reference)
    half = frame.iloc[len(frame)//2:]
    fixed = frame.status.isin(['fixed', 'partial_fixed']).mean() if 'status' in frame else 0.0
    return dict(solution=label, epochs=len(frame), reference=reference_source,
                final_e=frame.east.iloc[-1], final_n=frame.north.iloc[-1], final_u=frame.up.iloc[-1],
                final_3d=frame.error_3d.iloc[-1],
                rms_e=float(np.sqrt(np.mean(half.east**2))), rms_n=float(np.sqrt(np.mean(half.north**2))),
                rms_u=float(np.sqrt(np.mean(half.up**2))), rms_3d=float(np.sqrt(np.mean(half.error_3d**2))),
                convergence_s=convergence_time(frame, horizontal, vertical), fix_rate=float(fixed),
                mean_satellites=float(frame.satellites.mean()),
                mean_nis=float(frame.nis.mean()) if 'nis' in frame else np.nan)


def statistics_table(rows):
    return pd.DataFrame(rows)
