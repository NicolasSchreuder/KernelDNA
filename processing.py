import numpy as np

import pandas as pd

def train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Splits training matrix X and target vector y in training and test set
    """
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