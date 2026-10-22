import numpy as np
from gps_processing.constants import A, B

def elevation(receiver, satellites):
    normal = np.asarray(receiver)/np.array([A*A,A*A,B*B])
    normal /= np.linalg.norm(normal)
    delta = np.asarray(satellites)-receiver
    return np.degrees(np.arcsin(np.clip((delta@normal)/np.linalg.norm(delta,axis=1),-1,1)))

def initial_covariance(satellites=32):
    return np.diag([100.0**2]*3 + [1e5**2] + [20.0**2]*satellites)
