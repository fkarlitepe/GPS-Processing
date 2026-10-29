import numpy as np
from gps_processing.constants import C,F1,F2
LAMBDA_W=C/(F1-F2)
LAMBDA_N=C/(F1+F2)
BETA=C*F2/(F1**2-F2**2)

def combination_covariance(sine, code=3.0, phase=0.003, f1=F1, f2=F2):
    """Propagate independent raw P1,P2,L1,L2 metre noise, preserving cross terms
    (``f1``/``f2``: GPS L1/L2 by default, Galileo E1/E5a)."""
    transform=np.array([[f1**2/(f1**2-f2**2),-f2**2/(f1**2-f2**2),0,0],
        [0,0,f1**2/(f1**2-f2**2),-f2**2/(f1**2-f2**2)],
        [-f1/(f1+f2),-f2/(f1+f2),f1/(f1-f2),-f2/(f1-f2)]])
    return transform@np.diag([code**2,code**2,phase**2,phase**2])@transform.T/sine**2

def difference_matrix(count, reference):
    """Between-satellite differences; ``reference`` is one position or, with several systems,
    the reference position of every satellite (each system differenced to its own reference)."""
    of=np.full(count,reference) if np.ndim(reference)==0 else np.asarray(reference)
    others=[i for i in range(count) if of[i]!=i]
    d=np.zeros((len(others),count))
    for j,i in enumerate(others): d[j,i]=1; d[j,of[i]]=-1
    return np.kron(d,np.eye(3))
