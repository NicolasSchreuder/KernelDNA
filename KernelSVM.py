from cvxopt import matrix
from cvxopt import solvers

from kernel import build_kernel_vector, build_kernel_matrix, build_kernel_matrix_from_string, build_kernel_vector_from_string, spectrum_kernel, build_spectrum_kernel_matrix, build_spectrum_kernel_vector

from tqdm import tqdm

import numpy as np

class KernelSVM():
    
    def __init__(self, lambda_reg, kernel, kernel_parameters, data_type='vector', threshold=1e-3, verbose = 1):
        self.lambda_reg = lambda_reg # regularization parameter
        self.kernel = kernel # kernel choice
        self.data_type = data_type # vector or string
        self.kernel_parameters = kernel_parameters
        self.kernel_matrix_is_built = False
        self.threshold = threshold
        self.verbose = verbose
    
    def verbprint(self,*args):
        if self.verbose > 0:
            print(*args)
    
    def verbverbprint(self,*args):
        if self.verbose > 1:
            print(*args)
    
    def get_w(self):
        #Return w when the kernel is linear
        #w = sum(alpha * sv_y * sv for alpha, sv, sv_y in zip(self.alpha, self.sv, self.sv_y))
        #w  = sum(alpha * 1 * sv for alpha, sv, sv_y in zip(self.alpha, self.sv, self.sv_y))
        w = sum(self.alpha*self.sv)
        return w
        
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
        P = 2*K.astype(np.float64) 
        q = -2*y.astype(np.float64) 
        G = np.concatenate([np.diag(y), -np.diag(y)], axis=0).astype(np.float64) 
        h = (np.concatenate([np.ones(n), np.zeros(n)])/(2*self.lambda_reg*n)).astype(np.float64) 
        
        #print(P.shape, q.shape, G.shape, h.shape) 
        # Convert matrices and vectors to the right format for cvxopt solver
        # cf http://cvxopt.org/userguide/coneprog.html for solver's doc
        P_solver, q_solver = matrix(P), matrix(q)
        G_solver, h_solver = matrix(G), matrix(h)
        
        # Find solution using cvxopt
        sol = solvers.qp(P=P_solver, q=q_solver, G=G_solver, h=h_solver)
        alpha = np.array(sol['x'])
        self.alpha_old = alpha
        
        #Compute the suppor vectors, and discard those whose lagrange multipliers is < threshold
        
        self.sv_ind = np.where(np.abs(alpha) > self.threshold)[0] #indices of the support vectors
        #print(self.sv_ind)
        self.alpha = alpha[self.sv_ind]
        self.sv = X[self.sv_ind, :]
        self.sv_y = y[self.sv_ind]
        self.n_support = len(self.sv_ind)
        
        #Compute bias from KKT conditions, we use the average on the support vectors (instead of one evaluation) for stability.
        b = 0
        for n in range(self.n_support):
            b += self.sv_y[n]
            b -= sum(alpha * self.kernel(self.sv[n], sv,**self.kernel_parameters) for alpha, sv in zip(self.alpha, self.sv))
                     
        b = b/self.n_support
        self.b = b
        self.verbprint("numbers of support vectors : {}".format(self.n_support))
        self.verbverbprint("bias: {}".format(self.b))
        
    def project(self, X_test):
        y_predict = np.zeros(len(X_test))
        for i in range(len(X_test)):
            y_predict[i] = sum(alpha * self.kernel(X_test[i], sv,**self.kernel_parameters) for alpha, sv in zip(self.alpha, self.sv))
        
        return y_predict + self.b
    
    def predict(self, X_test):
        self.verbprint("  Predicting on a BinarySVC for data X_test of shape {} ...".format(np.shape(X_test)))
        predictions = np.sign(self.project(X_test))
        self.verbprint("  Stats about the predictions: (0 should never be predicted, labels are in {-1,+1})\n", list((k, np.sum(predictions == k)) for k in [-1, 0, +1]))
        
        return predictions
    
    
    def pred(self, X_test):
        
        if self.data_type == 'vector':
            n = X_test.shape[0]
            y_pred = np.zeros(n)
            for i in range(n):
                y_pred[i] = np.sign(np.dot(self.alpha_old.T, 
                                           build_kernel_vector(self.X_train, X_test[i, :], 
                                                               self.kernel, self.kernel_parameters)))
        elif self.data_type == 'string':
            n = len(X_test)
            y_pred = np.zeros(n)
            
            if self.kernel == spectrum_kernel:
                for i in tqdm(range(n), desc='Predicting values'):
                # self.phi_x corresponds to the previously computed representation of train data
                    y_pred[i] = np.sign(np.dot(self.alpha_old.T, build_spectrum_kernel_vector(self.X_train, X_test[i], self.kernel_parameters)))
                
            else:
                for i in range(n):
                    y_pred[i] = np.sign(np.dot(self.alpha_old.T, 
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