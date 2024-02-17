import numpy as np

#defining transformation functions
def logit(p: float):
    """Transforms value p, changing bounds from [0,1] to (-inf,inf). Inverse of sigmoid."""
    if not 0 <= p <= 1: raise ValueError("p must be bounded [0,1]!")
    return np.log(p/(1-p))

def sigmoid(x: float):
    """Transforms value x, changing bounds from (-inf,inf) to [0,1]. Inverse of logit."""
    return 1/(1+np.exp(-x))