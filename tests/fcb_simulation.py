"""Synthetic reference network for FCB/UFO estimator tests: b = n + r_i(t) + s_j(t) + e."""
import numpy as np
import pandas as pd


def network(stations=8, sats=14, blocks=48, interval=900.0, sat_walk=0.01, rec_walk=0.02,
            outliers=0.06, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-0.5, 0.5, sats)[None, :]+np.cumsum(rng.normal(0, sat_walk, (blocks, sats)), axis=0)
    r = rng.uniform(-0.5, 0.5, stations)[None, :]+np.cumsum(rng.normal(0, rec_walk, (blocks, stations)), axis=0)
    r[:, 0] -= r[:, 0].copy()                                   # truth expressed in the station-0 datum
    rows = []
    rise = rng.uniform(-blocks/2, blocks, (stations, sats))       # contiguous visibility windows
    for t in range(blocks):
        for i in range(stations):
            for j in range(sats):
                age = t-rise[i, j]
                if not 0 <= age < blocks/2.5:
                    continue
                elevation = 10+70*np.sin(np.pi*age/(blocks/2.5))
                formal = 0.03                                           # what the float filter reports
                noise = 0.02+0.06*(1-np.sin(np.radians(elevation)))       # true scatter grows at low elevation
                bias = 0.0
                if elevation < 25 and rng.random() < outliers*4:
                    bias = rng.choice([-1, 1])*rng.uniform(0.15, 0.35)    # multipath-like biased ambiguity
                value = rng.integers(-20, 20)+r[t, i]+s[t, j]+bias+rng.normal(0, noise)
                rows.append(dict(station=f'S{i:02d}', sat=j+1, block=t, epok=t*interval, value=value,
                                 sigma=formal, elevation=elevation, arc_epochs=30*(age+1)))
    truth = pd.DataFrame([dict(block=t, sat=j+1, ufo=s[t, j]) for t in range(blocks) for j in range(sats)])
    return pd.DataFrame(rows), truth


def satellite_error(result, truth):
    """RMS (cycles) of between-satellite UFO differences against the truth, modulo 1."""
    merged = result.satellite.merge(truth, on=['block', 'sat'], suffixes=('', '_true'))
    errors = []
    for _, g in merged.groupby('block'):
        if len(g) < 2:
            continue
        d = (g.ufo-g.ufo_true).to_numpy()
        centre = np.angle(np.exp(2j*np.pi*d).mean())/(2*np.pi)          # common datum offset
        errors.append(((d-centre)+0.5) % 1-0.5)
    e = np.concatenate(errors)
    return float(np.sqrt(np.mean(e**2)))
