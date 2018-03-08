from cvxopt import matrix
from cvxopt import solvers

from processing import build_kernel_vector, build_kernel_matrix, build_kernel_matrix_from_string, build_kernel_vector_from_string

import numpy as np

class KernelSVM():
    
    def __init__(self, lambda_reg, kernel, data_type='num'):
        self.lambda_reg = lambda_reg # regularization parameter
        self.kernel = kernel # kernel choice
        self.data_type = data_type
        
    def fit(self, X, y):
        
        # Save train matrix for computing kernel evaluation later
        self.X_train = X
        
        # Build kernel matrix
        if self.data_type == 'num':
            K = build_kernel_matrix(X, self.kernel)
        else:
            K = build_kernel_matrix_from_string(X, self.kernel)

        n = K.shape[0]
        
        # Dual SVM problem to generic CVXOPT QP
        P = 2*K
        q = -2*y
        G = np.concatenate([np.diag(y), -np.diag(y)], axis=0)
        h = np.concatenate([np.ones(n), np.zeros(n)])/(2*self.lambda_reg*n)
        
        # Convert matrices and vectors to the right format for cvxopt solver
        # cf http://cvxopt.org/userguide/coneprog.html for solver's doc
        P_solver, q_solver = matrix(P), matrix(q)
        G_solver, h_solver = matrix(G), matrix(h)

        sol = solvers.qp(P=P_solver, q=q_solver, G=G_solver, h=h_solver)
        self.alpha = np.array(sol['x'])
    
    def pred(self, X_test):
        if self.data_type == 'num':
            n = X_test.shape[0]
            y_pred = np.zeros(n)
            for i in range(n):
                y_pred[i] = np.sign(np.dot(self.alpha.T, 
                                           build_kernel_vector(self.X_train, X_test[i, :], 
                                                               self.kernel)))
        else:
            n = len(X_test)
            y_pred = np.zeros(n)
            for i in range(n):
                y_pred[i] = np.sign(np.dot(self.alpha.T, 
                                           build_kernel_vector_from_string(self.X_train, 
                                                                           X_test[i], self.kernel)))
        return y_pred
    
    
    def save(self, filename_alpha, filename_train):
        np.save(filename_alpha, self.alpha)
        np.save(filename_train, self.X_train)
        
    def load(self, filename_alpha, filename_train):
        self.alpha = np.load(filename_alpha)
        self.X_train = np.load(filename_train)