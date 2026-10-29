import numpy as np
import pandas as pd
from gps_processing.constants import C, OMEGA_E

def interpolate(nodes, values, target):
    """Local degree-nine barycentric Lagrange, with centred/scaled times."""
    if len(nodes) < 10:
        raise ValueError('Yörünge enterpolasyonu için en az 10 örnek gerekli.')
    # Up to 60 s beyond the table (signal travel time before the first node of a single-day file).
    if target < nodes[0]-60 or target > nodes[-1]+60:
        raise ValueError('Gözlem zamanı SP3 kapsamı dışında.')
    start = max(0, min(np.searchsorted(nodes, target)-5, len(nodes)-10))
    x = (nodes[start:start+10]-target)/900.0
    y = values[start:start+10]
    exact = np.flatnonzero(np.abs(x) < 1e-12)
    if len(exact):
        return y[exact[0]]
    weights = np.array([1/np.prod(x[k]-np.delete(x,k)) for k in range(10)])
    w = weights/(-x)
    return w @ y / w.sum()

def lagrange(nodes, values, targets, scale=900.0):
    """Vectorized form of ``interpolate``: degree-nine barycentric Lagrange at many ``targets``."""
    targets = np.asarray(targets, float)
    if len(nodes) < 10:
        raise ValueError('Yörünge enterpolasyonu için en az 10 örnek gerekli.')
    start = np.clip(np.searchsorted(nodes, targets)-5, 0, len(nodes)-10)
    window = start[:, None]+np.arange(10)[None, :]
    x = (nodes[window]-targets[:, None])/scale                       # (m, 10)
    difference = x[:, :, None]-x[:, None, :]
    difference[:, np.arange(10), np.arange(10)] = 1.0
    weights = 1/np.prod(difference, axis=2)
    exact = np.abs(x) < 1e-12
    safe = np.where(exact, 1.0, x)
    w = weights/(-safe)
    y = values[window]                                                # (m, 10, 3)
    out = np.einsum('mk,mkd->md', w, y)/w.sum(axis=1)[:, None]
    hit = exact.any(axis=1)
    if hit.any():
        out[hit] = y[hit, np.argmax(exact[hit], axis=1)]
    return out


def interpolate_satellites(orbit, observations, clocks=None):
    """``clocks``: optional high-rate clock interpolator (io.clock_rinex.Clocks); observations of a
    satellite/epoch the clock file does not cover are dropped (counted in attrs['orbit_clock_dropped']).

    Vectorized per satellite (same results as the scalar ``interpolate``)."""
    groups = {int(s): f.sort_values('epok') for s,f in orbit.groupby('sat')}
    parts = []
    for sat, obs in observations.groupby('sat', sort=False):
        if int(sat) not in groups:
            continue                     # satellite absent from the orbit product (e.g. unhealthy)
        frame = groups[int(sat)]
        t = frame.epok.to_numpy(float)
        epochs = obs.epok.to_numpy(float)
        inside = (epochs >= t[0]) & (epochs <= t[-1]+30)
        if (~inside).any() and t[0] <= 0 and t[-1] >= 86400:
            raise ValueError('Gözlem zamanı SP3 kapsamı dışında.')
        epochs, travel = epochs[inside], obs.travel_code.to_numpy(float)[inside]/C
        if clocks is not None:
            clock = clocks.values(int(sat), epochs)
        else:
            clock = np.interp(epochs, t, frame.time.to_numpy())
        ok = np.isfinite(clock)
        epochs, travel, clock = epochs[ok], travel[ok], clock[ok]
        emission = epochs-travel-clock*1e-6
        if len(emission) and (emission.min() < t[0]-60 or emission.max() > t[-1]+60):
            raise ValueError('Gözlem zamanı SP3 kapsamı dışında.')
        nodes = frame[['x','y','z']].to_numpy()
        xyz = lagrange(t, nodes, emission)*1000
        # Central difference velocity (m/s) for the periodic relativistic clock term.
        velocity = (lagrange(t, nodes, emission+.5)-lagrange(t, nodes, emission-.5))*1000
        relativity = -2*np.einsum('ij,ij->i', xyz, velocity)/C
        angle = OMEGA_E*travel
        cos, sin = np.cos(angle), np.sin(angle)
        xyz = np.column_stack([cos*xyz[:, 0]+sin*xyz[:, 1], -sin*xyz[:, 0]+cos*xyz[:, 1], xyz[:, 2]])
        if clocks is not None:
            clock = clocks.values(int(sat), emission)
        else:
            clock = np.interp(emission, t, frame.time.to_numpy())
        ok = np.isfinite(clock)
        parts.append(pd.DataFrame(dict(epok=epochs[ok], sat=int(sat), t_ems=emission[ok], x=xyz[ok, 0], y=xyz[ok, 1],
                                       z=xyz[ok, 2], dt_sat=clock[ok], relativity_m=relativity[ok])))
    columns = ['epok','sat','t_ems','x','y','z','dt_sat','relativity_m']
    satellites = pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
    satellites = satellites.astype({'sat': observations.sat.dtype})
    result = observations.merge(satellites[['epok','sat','t_ems','dt_sat']], on=['epok','sat'], validate='one_to_one')
    result.attrs.update(observations.attrs)
    result.attrs['orbit_clock_dropped'] = len(observations)-len(result)
    satellites = satellites.sort_values(['epok', 'sat'], kind='stable').reset_index(drop=True)
    return result, satellites
