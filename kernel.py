"""Manual Implementation of various kernels for Euclidean datas (i.e vectors)
Each kernel function takes as input two arrays X,Y of sizes (nx1) and (mx1) and outputs the Kernel Matrix Kxy = K(X,Y) of size (nxn)"""

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from onehotdna import sequence_to_matrix

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
    """
    Returns k-spectrum for a given string
    """
    spectrum = [string[i:i+k] for i in range(len(string)-k+1)]
    return spectrum

def spectrum_kernel(x, y, kernel_parameters):
    """
    k-spectrum kernel evaluation between string x and y
    """
    k = kernel_parameters['k']
    K_xy = 0
    
    # We store the spectrum as Counter dictionaries for efficient computing
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
    
    # Compute k-spectrum on training data
    phi_X = []
    for i in range(n):
        phi_X.append(Counter(get_spectrum(X_train[i], k)))
    
    # Compute k-spectrum on test point
    phi = Counter(get_spectrum(x, k))
    
    # Build kernel vector for test point
    K_x = np.zeros(n)
    for i in range(n):
        K_x[i] = sum([phi[key]*phi_X[i][key]
                       for key in phi.keys() if key in phi_X[i].keys()])
    return K_x


#### Implementation of "Convolutional Kitchen Sinks"
### as described in http://vaishaal.com/genomics_kitchensinks.pdf

def get_k_grams(x, k, alpha_size=4):
    """
    Decomposes a 1-d array into k-grams
    """
    outsize = int(x.shape[0]/alpha_size - k + 1) # output size
    kgrams = np.zeros((outsize, int(k*alpha_size))) # matrix of k-grams
    for i in range(outsize):
        kgrams[i] = x[i*alpha_size:(i+k)*alpha_size]
    return kgrams

def compute_conv_features(X, W, b, alpha_size=4):
    """
    Computes convulutional features for kitchen sink
    Input : 
            X : should be the ouput of sequence_to_matrix(X_raw), size (num_samples, 4*d = n)
            W :  Array of Gaussian vectors of size (M,4*k), W_ij should be drawn iid N(O,gamma)
            b : vector of size M, should be drawn uniformly on [0,2Pi]
        Output :
            Matrix of feature vectors : (num_samples,M)
     """
    num_samples, d = X.shape[0], int(X.shape[-1]/4)
    M = W.shape[0]   
    k = int(W.shape[-1]/4)     
    X_lift = np.zeros((num_samples, M)) 
    scale = np.sqrt(2/float(W.shape[0])) 
    for i in range(X.shape[0]):
        kgrams = get_k_grams(X[i,:], k, alpha_size) #size (d-k+1, 4k)
        xlift_conv = np.cos(np.dot(kgrams, W.T) + b) #size (d-k+1, M)
        X_lift[i] = scale*np.sum(xlift_conv, axis=0)
    return X_lift

def phi_sink(X_raw, k, gamma, M, alpha_size=4):
    """
    Input:
        X_raw : list of DNA (or any string) sequence
        k : k-gram parameter
        gamma : Width of the kernel
        M : number of kitchen sinks
    Output:
        Matrix of features vectors
    """
    # Convert list of string to DNA one-hot encoding vectorial representation
    X = sequence_to_matrix(X_raw) 
    
    # Generate random vectors for stochastic approximation
    W = gamma*np.random.randn(M,4*k)
    b = (2*np.pi)*np.random.rand(M)
    
    # Compute features on vectorial representation of sequences
    X_lift = compute_conv_features(X, W, b, alpha_size = alpha_size)

    return X_lift