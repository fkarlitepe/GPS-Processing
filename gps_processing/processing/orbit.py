import numpy as np
import pandas as pd
from gps_processing.constants import C, OMEGA_E

def interpolate(nodes, values, target):
    """Local degree-nine barycentric Lagrange, with centred/scaled times."""
    if len(nodes) < 10:
        raise ValueError('Yörünge enterpolasyonu için en az 10 örnek gerekli.')
    if target < nodes[0] or target > nodes[-1]:
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

def interpolate_satellites(orbit, observations):
    groups = {int(s): f.sort_values('epok') for s,f in orbit.groupby('sat')}
    rows = []
    for obs in observations.itertuples(index=False):
        if obs.sat not in groups:
            raise ValueError(f'G{obs.sat:02d} için SP3 kaydı yok.')
        frame = groups[obs.sat]
        t = frame.epok.to_numpy()
        if obs.epok < t[0] or obs.epok > t[-1]:
            raise ValueError('Gözlem zamanı SP3 kapsamı dışında.')
        clock = np.interp(obs.epok, t, frame.time.to_numpy())
        travel = obs.travel_code/C
        emission = obs.epok-travel-clock*1e-6
        nodes = frame[['x','y','z']].to_numpy()
        xyz = interpolate(t, nodes, emission)*1000
        # Central difference velocity (m/s) for the periodic relativistic clock term.
        velocity = (interpolate(t, nodes, min(emission+.5, t[-1]))-interpolate(t, nodes, max(emission-.5, t[0])))*1000
        relativity = -2*xyz@velocity/C
        angle = OMEGA_E*travel
        rotation = np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
        xyz = rotation @ xyz
        clock = np.interp(emission, t, frame.time.to_numpy())
        rows.append((obs.epok, obs.sat, emission, *xyz, clock, relativity))
    satellites = pd.DataFrame(rows, columns=['epok','sat','t_ems','x','y','z','dt_sat','relativity_m'])
    result = observations.merge(satellites[['epok','sat','t_ems','dt_sat']], on=['epok','sat'], validate='one_to_one')
    result.attrs.update(observations.attrs)
    return result, satellites
