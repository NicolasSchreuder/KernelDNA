import numpy as np

from cvxopt import matrix
from cvxopt import solvers

solvers.options['show_progress'] = False # quiet solver (no print)

def SVM_transform_Xy_to_QP(X, y, tau):
    """
    Transform SVM problem with feature matrix X and label vector y to a QP with linear constraint :
    """
    n, d = X.shape

    Q = np.block([
        [np.eye(d), np.zeros((d, n))],
        [np.zeros((n, d+n))]
    ])

    p = 1/(tau*n)*np.concatenate((np.zeros(d), np.ones(n)))

    A_1 = np.concatenate((-np.dot(np.diag(np.squeeze(np.array(y))), X), 
                          -np.eye(n)), axis=1)
    A_2 = np.concatenate((np.zeros((n, d)), -np.eye(n)), axis=1)

    A  = np.concatenate((A_1, A_2), axis=0)

    b = np.concatenate((-np.ones(n), np.zeros(n)))

    return Q, p, A, b

class SVM:
    """
    Implementation of a Support Vector Machine
    """
    def __init__(self, tau):
        self.w = None
        self.tau = tau # regularization parameter 
    
    def fit(self, X, y):
        n, d = X.shape
        
        # Transforms the problem into a QP under linear contraints
        Q, p, A, b = SVM_transform_Xy_to_QP(X, y, self.tau)
        
        # Converts to the right format for cvxopt solver
        Q_solver, p_solver = matrix(Q), matrix(p)
        A_solver, b_solver = matrix(A), matrix(b)

        
        sol = solvers.qp(Q_solver,p_solver,A_solver,b_solver)
        self.w = np.array(sol['x'][:d])
    
    def pred(self, X):
        return np.sign(np.dot(X, self.w))
    
    def get_weights(self):
        return self.w
   
    def score(self, X, y):
        
        y_pred = SVM.pred(self, X)
        y_pred = np.reshape(y_pred, y.shape)
        return 1 - np.mean(np.abs(y-y_pred)/2)
