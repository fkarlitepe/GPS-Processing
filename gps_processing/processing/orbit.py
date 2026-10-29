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

def interpolate_satellites(orbit, observations, clocks=None):
    """``clocks``: optional high-rate clock interpolator (io.clock_rinex.Clocks); observations of a
    satellite/epoch the clock file does not cover are dropped (counted in attrs['orbit_clock_dropped'])."""
    groups = {int(s): f.sort_values('epok') for s,f in orbit.groupby('sat')}
    rows = []
    for obs in observations.itertuples(index=False):
        if obs.sat not in groups:
            continue                     # satellite absent from the orbit product (e.g. unhealthy)
        frame = groups[obs.sat]
        t = frame.epok.to_numpy()
        if obs.epok < t[0] or obs.epok > t[-1]+30:
            if len(t) and t[0] <= 0 and t[-1] >= 86400:
                raise ValueError('Gözlem zamanı SP3 kapsamı dışında.')
            continue                     # single-day orbit: epochs past its last node are dropped
        if clocks is not None and not clocks.covers(obs.sat, obs.epok):
            continue
        clock = clocks(obs.sat, obs.epok) if clocks is not None else np.interp(obs.epok, t, frame.time.to_numpy())
        travel = obs.travel_code/C
        emission = obs.epok-travel-clock*1e-6
        nodes = frame[['x','y','z']].to_numpy()
        xyz = interpolate(t, nodes, emission)*1000
        # Central difference velocity (m/s) for the periodic relativistic clock term.
        velocity = (interpolate(t, nodes, emission+.5)-interpolate(t, nodes, emission-.5))*1000
        relativity = -2*xyz@velocity/C
        angle = OMEGA_E*travel
        rotation = np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
        xyz = rotation @ xyz
        if clocks is not None:
            if not clocks.covers(obs.sat, emission):
                continue
            clock = clocks(obs.sat, emission)
        else:
            clock = np.interp(emission, t, frame.time.to_numpy())
        rows.append((obs.epok, obs.sat, emission, *xyz, clock, relativity))
    satellites = pd.DataFrame(rows, columns=['epok','sat','t_ems','x','y','z','dt_sat','relativity_m'])
    result = observations.merge(satellites[['epok','sat','t_ems','dt_sat']], on=['epok','sat'], validate='one_to_one')
    result.attrs.update(observations.attrs)
    result.attrs['orbit_clock_dropped'] = len(observations)-len(result)
    return result, satellites
