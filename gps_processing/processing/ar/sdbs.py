"""Between-satellite differences with full induced observation covariance."""
from .models import difference_matrix

def transform(y, predicted, design, noise, reference):
    d=difference_matrix(len(y)//3,reference)
    return d@y,d@predicted,d@design,d@noise@d.T
