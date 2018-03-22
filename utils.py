import csv
import numpy as np

e1,e2,e3,e4 = np.eye(4)
ATCG_MAP = {'A': e1, 'T': e2, 'C': e3, 'G': e4}

def sequence_to_matrix(X_raw):
    """Input :
            X_raw is supposed to be an array of n DNA sequences of sizes d 
       Output :
            X the (n,4*d) real matrix, X=ATCG_MAP(X_raw)      """

    n,d = len(X_raw), len(X_raw[0])
    X = np.zeros((n,4*d))
    for i in range(n):
        x_i_raw = X_raw[i]
        x_i = np.concatenate([ ATCG_MAP[x] for x in x_i_raw]) # A is represented by e1, T by e2 ...
        X[i] = x_i
    return X

    






