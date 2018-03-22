"""Manual Implementation of various kernels for euclidean datas (i.e vectors)
Each kernel function takes as input two arrays X,Y of sizes (nx1) and (mx1) and outputs the Kernel Matrix Kxy = K(X,Y) of size (nxm)"""

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

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
    """ Linear kernel : dot product K(X, Y) = (X^T) Y
    """
    return np.dot(X, Y)

# Compute matrices and vectors for a given kernel

def build_kernel_matrix(X, pd_kernel, kernel_parameters):
    """
    Builds kernel matrix (K(x_i, x_j))_(i,j) given numeric training data and a kernel
    X : training data matrix (Numpy array)
    pdf_kernel : a positive definite kernel (function)
    """
    n, d = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = pd_kernel(X[i, :], X[j, :], **kernel_parameters)
            K[j, i] = K[i, j]
    return K

def build_kernel_vector(X, x, pd_kernel, kernel_parameters):
    """
    Builds kernel vector (K(x, x_i))_i for a given vector x and training data
    X : training data matrix (Numpy array)
    x : test point
    pdf_kernel : a positive definite kernel (function)
    """
    n, d = X.shape
    K_x = np.zeros(n)
    for i in range(n):
        K_x[i] = pd_kernel(X[i, :], x, **kernel_parameters)
    return K_x

def build_kernel_vector_from_string(X, x, pd_kernel, kernel_parameters):
    """
    Builds kernel vector (K(x, x_i))_i for a given vector x and training data
    X : training data matrix (Numpy array)
    x : test point
    pdf_kernel : a positive definite kernel (function)
    """
    n = len(X)
    K_x = np.zeros(n)
    for i in range(n):
        K_x[i] = pd_kernel(X[i], x, **kernel_parameters)
    return K_x

def build_kernel_matrix_from_string(X, kernel_parameters):
    """
    Builds kernel matrix (K(x_i, x_j))_(i,j) given string training data and a kernel
    X : training data matrix (list of string)
    pdf_kernel : a positive definite kernel (function)
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = pd_kernel(X[i], X[j], **kernel_parameters)
            K[j, i] = K[i, j]
    return K


# Spectrum kernel

from collections import Counter

def get_spectrum(string, k=3):
    spectrum = [string[i:i+k] for i in range(len(string)-k+1)]
    return spectrum

def spectrum_kernel(x, y, kernel_parameters):
    """
    Spectrum kernel for string data
    """
    k = kernel_parameters['k']
    K_xy = 0
    phi_x = Counter(get_spectrum(x, k))
    phi_y = Counter(get_spectrum(y, k))
    for key in phi_x.keys():
        if key in phi_y.keys():
            K_xy += phi_x[key]*phi_y[key]
    return K_xy


from mismatchtree import mismatchTree, inMismatchTree

def build_spectrum_kernel_matrix(X, kernel_parameters):
    """
    Builds kernel matrix (K(x_i, x_j))_(i,j) given string training data for spectrum kernel
    X : training data matrix (list of string)
    pdf_kernel : a positive definite kernel (function)
    """
    k = kernel_parameters['k'] # spectrum kernel parameter
    n = len(X)
    K = np.zeros((n, n))

    # Get spectrum for each string
    spectrum = []
    for i in range(n):
            spectrum.append(Counter(get_spectrum(X[i], k)))

    mismatch_trees = {}
    in_mismatch_trees = {}

    if 'm' in kernel_parameters.keys(): # allow mismatch of size m
        m = kernel_parameters['m'] # mismatch parameters
        for i in tqdm(range(n), desc='Building kernel matrix'):
            for j in range(i+1):
                for key in spectrum[i]:

                    # check if mismatch tree has already been computed
                    # if not, it is computed and stored inside the dictionary
                    if key not in mismatch_trees:
                        mismatch_trees[key] = mismatchTree(key, m+1)

                    # Check if correspondence between mismatch tree
                    # and list of keys has already been computed
                    if (j, key) not in in_mismatch_trees:
                        in_mismatch_trees[j, key] = inMismatchTree(mismatch_trees[key], spectrum[j])

                    K[i, j] += spectrum[i][key] * sum([spectrum[j][mismatch_key]
                                  for mismatch_key in in_mismatch_trees[j, key]])
    
                K[j, i] = K[i, j] # symmetric matrix
                    
                if i==j:
                    K[i, i] *= 48 
                        
    else: # no mismatch allowed
        for i in tqdm(range(n), desc='Building kernel matrix'):
            for j in range(i+1):
                K[i, j] = sum([spectrum[i][key]*spectrum[j][key]
                           for key in spectrum[i] if key in spectrum[j]])
                K[j, i] = K[i, j]

    return K


def build_spectrum_kernel_vector(X_train, x, kernel_parameters):
    """
    Builds kernel vector (K(x, x_i))_i for a given vector x and training data
    X : training data matrix (Numpy array)
    x : test point
    """
    n = len(X_train)
    k = kernel_parameters['k']
    phi_X = []
    for i in range(n):
        phi_X.append(Counter(get_spectrum(X_train[i], k)))
    K_x = np.zeros(n)
    k = kernel_parameters['k']
    phi = Counter(get_spectrum(x, k))
    for i in range(n):
        K_x[i] = sum([phi[key]*phi_X[i][key]
                       for key in phi.keys() if key in phi_X[i].keys()])
    return K_x
