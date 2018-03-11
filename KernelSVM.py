from cvxopt import matrix
from cvxopt import solvers

from kernel import build_kernel_vector, build_kernel_matrix, build_kernel_matrix_from_string, build_kernel_vector_from_string, spectrum_kernel, build_spectrum_kernel_matrix, build_spectrum_kernel_vector

from tqdm import tqdm

import numpy as np

class KernelSVM():
    
    def __init__(self, lambda_reg, kernel, kernel_parameters, data_type='vector'):
        self.lambda_reg = lambda_reg # regularization parameter
        self.kernel = kernel # kernel choice
        self.data_type = data_type # vector or string
        self.kernel_parameters = kernel_parameters
        self.kernel_matrix_is_built = False
        
    def fit(self, X, y):
        
        # Save train matrix for computing kernel evaluation later
        self.X_train = X
                
        if not self.kernel_matrix_is_built: 
            # Build kernel matrix
            if self.data_type == 'vector':
                K = build_kernel_matrix(X, self.kernel, self.kernel_parameters)

            elif self.data_type == 'string' :
                if self.kernel == spectrum_kernel:
                    K = build_spectrum_kernel_matrix(X, self.kernel_parameters)

            else:
                raise "Data type not understood, should be either 'string' or 'vector' "
            
            self.K = K
            self.kernel_matrix_is_built = True
            
        else:
            K = self.K
        
        n = K.shape[0]
                
        # Convert dual SVM problem to generic CVXOPT quadratic program (cf KernelSVM notebook)
        P = 2*K
        q = -2*y
        G = np.concatenate([np.diag(y), -np.diag(y)], axis=0)
        h = np.concatenate([np.ones(n), np.zeros(n)])/(2*self.lambda_reg*n)
        
        # Convert matrices and vectors to the right format for cvxopt solver
        # cf http://cvxopt.org/userguide/coneprog.html for solver's doc
        P_solver, q_solver = matrix(P), matrix(q)
        G_solver, h_solver = matrix(G), matrix(h)
        
        # Find solution using cvxopt
        sol = solvers.qp(P=P_solver, q=q_solver, G=G_solver, h=h_solver)
        self.alpha = np.array(sol['x'])
    
    def pred(self, X_test):
        
        if self.data_type == 'vector':
            n = X_test.shape[0]
            y_pred = np.zeros(n)
            for i in range(n):
                y_pred[i] = np.sign(np.dot(self.alpha.T, 
                                           build_kernel_vector(self.X_train, X_test[i, :], 
                                                               self.kernel, self.kernel_parameters)))
        elif self.data_type == 'string':
            n = len(X_test)
            y_pred = np.zeros(n)
            
            if self.kernel == spectrum_kernel:
                for i in tqdm(range(n), desc='Predicting values'):
                # self.phi_x corresponds to the previously computed representation of train data
                    y_pred[i] = np.sign(np.dot(self.alpha.T, build_spectrum_kernel_vector(self.X_train, X_test[i], self.kernel_parameters)))
                
            else:
                for i in range(n):
                    y_pred[i] = np.sign(np.dot(self.alpha.T, 
                                               build_kernel_vector_from_string(self.X_train, 
                                                                               X_test[i], self.kernel, self.kernel_parameters)))
        
        else:
            raise "Data type not understood, should be either 'string' or 'vector'"
            
        return y_pred
    
    def save(self, filename_alpha, filename_train):
        np.save(filename_alpha, self.alpha)
        np.save(filename_train, self.X_train)
        
    def load(self, filename_alpha, filename_train):
        self.alpha = np.load(filename_alpha)
        self.X_train = np.load(filename_train)