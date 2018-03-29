from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False # no verbose for solver

from kernel import build_kernel_vector, build_kernel_matrix, build_kernel_matrix_from_string, build_kernel_vector_from_string, spectrum_kernel, build_spectrum_kernel_matrix, build_spectrum_kernel_vector

from tqdm import tqdm

import numpy as np

class KernelSVM():

    def __init__(self, loss, lambda_reg, kernel, kernel_parameters,
                    data_type='vector', threshold=1e-3, verbose = 1):
        """
        Initialization of a kernel SVM
        """
        self.lambda_reg = lambda_reg # regularization parameter
        self.kernel = kernel # kernel choice
        self.data_type = data_type # type of input data : vector or string
        self.kernel_parameters = kernel_parameters # optional parameters for the kernel
        self.kernel_matrix_is_built = False
        self.threshold = threshold # value threshold when selecting support vectors
        self.verbose = verbose
        self.loss = loss # type of loss : hinge or squared hinge

    def verbprint(self,*args):
        """
        Handle verbose
        """
        if self.verbose > 0:
            print(*args)

    def get_w(self):
        """Return w (weight vector) when the kernel is linear and the data vectorial"""
        if self.data_type == "vectorial":
            return sum(self.alpha*self.sv)
        else:
            raise "Could not compute w, data should be vectorial"

    def fit(self, X, y):
        """
        Train SVM on input features X and input target y
        """
        X = np.array(X)
        n = X.shape[0]

        if self.data_type=='vector': # if data is vectorial, add add dimension for bias
            X = np.concatenate((X, np.ones((n,1))), axis=1)

        # Store train matrix for computing kernel evaluation later
        self.X_train = X

        # Build the kernel matrix if it has not been built already
        if not self.kernel_matrix_is_built:

            if self.data_type == 'vector':
                K = build_kernel_matrix(X, self.kernel, self.kernel_parameters)

            elif self.data_type == 'string' :
                K = build_spectrum_kernel_matrix(X, self.kernel_parameters)

            else:
                raise "Data type not understood, should be either 'string' or 'vector' "

            # Store kernel matrix so we don't have to compute it again if needed
            self.K = K
            self.kernel_matrix_is_built = True

        else: # Load kernel matrix if it has already been built
            K = self.K

        # Convert dual SVM problem to generic CVXOPT quadratic program (cf KernelSVM notebook)

        if self.loss == 'hinge': # hinge loss : max(1-yf(x), 0)
            P = 2*K.astype(np.float64)
            q = -2*y.astype(np.float64)
            G = np.concatenate([np.diag(y), -np.diag(y)], axis=0).astype(np.float64)
            h = (np.concatenate([np.ones(n), np.zeros(n)])/(2*self.lambda_reg*n)).astype(np.float64)

        elif self.loss == 'squared_hinge': # squared hinge loss : max(1-yf(x), 0)**2
            P = 2*(K + n*self.lambda_reg*np.eye(n)).astype(np.float64)
            q = -2*y.astype(np.float64)
            G =  -np.diag(y).astype(np.float64)
            h =  np.zeros(n).astype(np.float64)

        else:
            raise "{} loss not implemented".format(loss)

        # Convert matrices and vectors to the right format for cvxopt solver
        # cf http://cvxopt.org/userguide/coneprog.html for solver's doc
        P_solver, q_solver = matrix(P), matrix(q)
        G_solver, h_solver = matrix(G), matrix(h)

        # Find solution using cvxopt
        sol = solvers.qp(P=P_solver, q=q_solver, G=G_solver, h=h_solver)
        alpha = np.array(sol['x'])
        self.alpha_old = alpha

        # Find the suppor vectors, and discard those whose lagrange multipliers is < threshold
        self.sv_ind = np.where(np.abs(alpha) > self.threshold)[0].astype(int) # indices of the support vectors
        self.alpha = alpha[self.sv_ind] # alpha coefficients corresponding to support vectors
        self.sv = X[self.sv_ind][:] # support vectors
        self.sv_y = y[self.sv_ind] # target corresponding to support vectors
        self.n_support = len(self.sv_ind) # number of support vectors

        self.verbprint("Numbers of support vectors : {}".format(self.n_support))

    def pred(self, X_test):

        if self.data_type == 'vector': # vectorial data
            n = X_test.shape[0] # number of observations in test set
            y_pred = np.zeros(n) # prediction vector

            X_test = np.concatenate((X_test, np.ones((n,1))), axis=1)
            #X_test = np.concatenate((X_test, np.ones((n,1))), axis=1) add one dimension for bias

            for i in tqdm(range(n), desc='Predicting values'):
                y_pred[i] = np.sign(sum(alpha * self.kernel(X_test[i], sv,**self.kernel_parameters)
                               for alpha, sv in zip(self.alpha, self.sv)))

        elif self.data_type == 'string': # string data
            n = len(X_test)
            y_pred = np.zeros(n) # prediction vector

            for i in tqdm(range(n), desc='Predicting values'):
                # self.phi_x corresponds to the previously computed representation of train data
                y_pred[i] = np.sign(np.dot(self.alpha_old.T,
                build_spectrum_kernel_vector(self.X_train,
                X_test[i], self.kernel_parameters)))

        else:
            raise "Data type not understood, should be either 'string' or 'vector'"

        return y_pred
