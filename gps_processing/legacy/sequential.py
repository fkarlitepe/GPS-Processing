"""v2.0 baseline (frozen): sequential float code/phase estimator, receiver clock and per-satellite
ambiguities, GPS only. Kept unchanged for regression tests and the method comparison."""
import numpy as np
import pandas as pd
from gps_processing.constants import C
from gps_processing.processing.geometry import elevation, initial_covariance

def solve(observations, satellites, approximate_xyz, mask=10, covariance=None):
    data = observations.merge(satellites[['epok','sat','x','y','z']], on=['epok','sat'], validate='one_to_one')
    state = np.zeros(36)
    state[:3] = approximate_xyz
    cov = initial_covariance() if covariance is None else np.asarray(covariance).copy()
    if cov.shape != (36,36):
        raise ValueError('Öncül kovaryans boyutu 36 × 36 olmalı.')
    positions, ambiguities, diagnostics, previous = [], [], [], {}
    last_correction = np.zeros(36)
    for epoch, frame in data.groupby('epok', sort=True):
        frame = frame.copy()
        frame['elevation'] = elevation(state[:3], frame[['x','y','z']].to_numpy())
        frame = frame[frame.elevation >= mask]
        if len(frame) < 4:
            diagnostics.append((epoch, len(frame), 'yetersiz uydu', np.nan))
            continue
        # Clock is independent at each epoch; ambiguity resets on flags, gaps or MW jump.
        cov[3,:] = 0; cov[:,3] = 0; cov[3,3] = 1e10
        for row in frame.itertuples():
            prior = previous.get(row.sat)
            if row.epoch_reset or prior is None or epoch-prior[0] > 120 or abs(row.A4-prior[1]) > 5:
                k = 3+int(row.sat)
                state[k] = row.Lc-row.Pc
                cov[k,:] = 0; cov[:,k] = 0; cov[k,k] = 1e6
            previous[row.sat] = (epoch,row.A4)
        prior_state, prior_cov = state.copy(), cov.copy()
        coords = frame[['x','y','z']].to_numpy()
        ids = frame.sat.to_numpy(dtype=int)+3
        size = len(frame)
        weight = np.maximum(np.sin(np.radians(frame.elevation.to_numpy())),0.1)
        noise = np.r_[ (3/weight)**2, (0.003/weight)**2 ]
        measured = np.r_[frame.Pc.to_numpy(), frame.Lc.to_numpy()]
        clock = C*frame.dt_sat.to_numpy()*1e-6
        for _ in range(8):
            delta = state[:3]-coords
            distance = np.linalg.norm(delta,axis=1)
            h = np.zeros((2*size,36))
            h[:size,:3] = delta/distance[:,None]
            h[size:,:3] = h[:size,:3]
            h[:,3] = 1
            h[size+np.arange(size), ids] = 1
            predicted_code = distance+state[3]-clock
            predicted = np.r_[predicted_code,predicted_code+state[ids]]
            innovation = measured-predicted+h@(state-prior_state)
            s = h@prior_cov@h.T+np.diag(noise)
            gain = np.linalg.solve(s,h@prior_cov).T
            updated = prior_state+gain@innovation
            change = np.linalg.norm(updated[:3]-state[:3])
            state = updated
            if change < 1e-4:
                break
        # Joseph form preserves symmetry/positive covariance under rounding.
        identity = np.eye(36)-gain@h
        cov = identity@prior_cov@identity.T+(gain*noise)@gain.T
        last_correction = state-prior_state
        ranges = np.linalg.norm(state[:3]-coords,axis=1)+state[3]-clock
        rms = float(np.sqrt(np.mean((frame.Pc.to_numpy()-ranges)**2)))
        positions.append((epoch,*state[:3],state[3]/C,size,rms))
        ambiguities.extend((epoch,int(sat),state[3+int(sat)]) for sat in frame.sat)
        diagnostics.append((epoch,size,'çözüldü',rms))
    if not positions:
        raise ValueError('Çözülebilen epok yok: yükseklik maskesini ve gözlem verilerini kontrol edin.')
    return (pd.DataFrame(positions,columns=['epok','x','y','z','receiver_clock_s','satellites','code_rms_m']),
            pd.DataFrame(ambiguities,columns=['epok','sat','N3']),
            pd.DataFrame(diagnostics,columns=['epok','satellites','status','code_rms_m']),cov,last_correction)
