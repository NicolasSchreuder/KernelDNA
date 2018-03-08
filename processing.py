import numpy as np

import pandas as pd

def train_test_split(X, y, test_size=0.3, random_state=42):
    
    np.random.seed(random_state) # for reproducibility
    
    n, d = X.shape
    
    nb_test_observations = int(test_size*n) # compute number of observations in future test set
    assert  nb_test_observations < n
    
    # Random permutation on the indices
    indices = np.random.permutation(n)
    
    training_idx, test_idx = indices[:nb_test_observations], indices[nb_test_observations:]
    X_train, X_test = X[training_idx, :], X[test_idx, :]
    y_train, y_test = y[training_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def load_mat50(path):
    """
    Load feature matrices
    """
    return np.loadtxt(open(path, "rb"), delimiter=" ", skiprows=0)

def load_y(path):
    """
    Load target vector 
    """
    return np.array(pd.read_csv(path, sep=',', index_col=0))

def build_kernel_matrix(X, pd_kernel):
    """
    Builds kernel matrix (K(x_i, x_j))_(i,j) given numeric training data and a kernel
    X : training data matrix (Numpy array)
    pdf_kernel : a positive definite kernel (function)
    """
    n, d = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = pd_kernel(X[i, :], X[j, :])
            K[j, i] = K[i, j]
    return K

def build_kernel_matrix_from_string(X, pd_kernel):
    """
    Builds kernel matrix (K(x_i, x_j))_(i,j) given string training data and a kernel
    X : training data matrix (list of string)
    pdf_kernel : a positive definite kernel (function)
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            K[i, j] = pd_kernel(X[i], X[j])
            K[j, i] = K[i, j]
    return K

def build_kernel_vector(X, x, pd_kernel):
    """
    Builds kernel vector (K(x, x_i))_i for a given vector x and training data 
    X : training data matrix (Numpy array)
    x : test point
    pdf_kernel : a positive definite kernel (function)
    """
    n, d = X.shape
    K_x = np.zeros(n)
    for i in range(n):
        K_x[i] = pd_kernel(X[i, :], x)
    return K_x

def build_kernel_vector_from_string(X, x, pd_kernel):
    """
    Builds kernel vector (K(x, x_i))_i for a given vector x and training data 
    X : training data matrix (Numpy array)
    x : test point
    pdf_kernel : a positive definite kernel (function)
    """
    n = len(X)
    K_x = np.zeros(n)
    for i in range(n):
        K_x[i] = pd_kernel(X[i], x)
    return K_x