import numpy as np
from gps_processing.constants import C,F1,F2
LAMBDA_W=C/(F1-F2)
LAMBDA_N=C/(F1+F2)
BETA=C*F2/(F1**2-F2**2)

def combination_covariance(sine, code=3.0, phase=0.003):
    """Propagate independent raw P1,P2,L1,L2 metre noise, preserving cross terms."""
    transform=np.array([[F1**2/(F1**2-F2**2),-F2**2/(F1**2-F2**2),0,0],
        [0,0,F1**2/(F1**2-F2**2),-F2**2/(F1**2-F2**2)],
        [-F1/(F1+F2),-F2/(F1+F2),F1/(F1-F2),-F2/(F1-F2)]])
    return transform@np.diag([code**2,code**2,phase**2,phase**2])@transform.T/sine**2

def difference_matrix(count, reference):
    others=[i for i in range(count) if i != reference]
    d=np.zeros((len(others),count))
    for j,i in enumerate(others): d[j,i]=1; d[j,reference]=-1
    return np.kron(d,np.eye(3))
