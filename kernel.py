"""Manual Implementation of various kernels for euclidean datas (i.e vectors)
Each kernel function takes as input two arrays X,Y of sizes (nx1) and (mx1) and outputs the Kernel Matrix Kxy = K(X,Y) of size (nxm)"""

import numpy as np
from numpy.linalg import norm

def rbf(X, Y, gamma='auto'):
    """ RBF t.i. kernel (radial base function), K(X, Y) = exp(- gamma * |X - Y|^2)

    - Give gamma = 'auto' or gamma = 0 to select gamma as 1 / n_features.
    """
    if gamma == 'auto' or gamma == 0:
        n_features = np.shape(X)[1] if len(np.shape(X)) > 1 else 1
        gamma = 1 / n_features
    assert gamma > 0, "[ERROR] kernels.rbf: using a gamma < 0 will not do what you want."
    return np.exp(- gamma * norm(X - Y)**2)

def laplace(X, Y, gamma='auto'):
    """ Laplace t.i. kernel, K(X, Y) = exp(- |X - Y| / gamma).

    - Give gamma = 'auto' or gamma = 0 to select gamma as 1 / n_features.
    """
    if gamma == 'auto' or gamma == 0:
        n_features = np.shape(X)[1] if len(np.shape(X)) > 1 else 1
        gamma = 1 / n_features
    assert gamma > 0, "[ERROR] kernels.laplace: using a gamma < 0 will not do what you want."
    return np.exp(- gamma * norm(X - Y))

def poly(X, Y, degree=3, coef0=1):
    """ Parametrized version of the polynomial kernel, K(X, Y) = (X . Y + coef0)^degree.

    - Default coef0 is 1.
    - Default degree is 3. Computation time is CONSTANT with d, but that's not a reason to try huge values. degree = 2,3,4,5 should be enough.
    - Using degree = 1 is giving a (possibly) non-homogeneous linear kernel.
    """
    assert degree > 0, "[ERROR] kernels.poly: using a degree < 0 will fail (the kernel is not p.d.)."
    return (np.dot(X, Y) + coef0) ** degree

def linear(X, Y):
    """ Linear kernel, simply the dot product K(X, Y) = (X . Y).
    """
    return np.dot(X, Y)  


# Spectrum kernel 

from collections import Counter

def get_spectrum(string, k=3):
    spectrum = [string[i:i+k] for i in range(len(string)-k+1)]
    return spectrum

def spectrum_kernel(x, y, k=3):
    """
    Spectrum kernel for string data
    """
    K_xy = 0
    phi_x = Counter(get_spectrum(x, k))
    phi_y = Counter(get_spectrum(y, k))
    for key in phi_x.keys():
        if key in phi_y.keys():
            K_xy += phi_x[key]*phi_y[key]
    return K_xy
