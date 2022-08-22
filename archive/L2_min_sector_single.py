#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cvxpy as cp
import logging

from network import Network


def L2_min_sector(NNo, nr, eps):
    '''
    Reduced-order synthesis where l = 1 ONLY (i.e. single hidden layer) for NNo, NNr.
    
    Parameters
    ----------
    NNo: Network (network.py)
        Full-order network.
        
    nr: int
        Dimension of hidden layer in reduced-order network, NNr.
        
    eps: float
        Constant term in error bound: ||e||^2 = gamma*||x||^2 + eps. Returned as epsv.
    
    Returns
    -------
    NNr: Network (network.py)
        Reduced-order network.
        
    gamv: float
        Linear term in error bound: ||e||^2 = gamv*||x||^2 + epsv.
        
    epsv: float
        Constant term in error bound: ||e||^2 = gamv*||x||^2 + epsv. Equal to eps.
        
    status: str
        Solution status. "Returns one of optimal, infeasible, or unbounded (with or without suffix inaccurate)."
        See: https://www.cvxpy.org/api_reference/cvxpy.problems.html    
    '''
    
    shape_w0 = NNo.NN['W0'].shape
    shape_w1 = NNo.NN['W1'].shape
    
    n = shape_w1[1]
    m = shape_w1[0]
    nx = shape_w0[1]
    
    j1 = np.eye(nr)
    j2 = np.zeros((n-nr, nr))
    J = np.vstack([j1, j2])
    
    # Define matrix variables
    J = cp.Constant(J)
    Q1 = cp.Variable((n, n), PSD=True)  # Q1 made general, previously diagonal
    Q2 = cp.Variable((nr, nr), PSD=True)   # Q2 made general, previously diagonal
    M1 = J@Q2
    M2 = cp.Variable((nr, n), nonneg=True)
    Fb = cp.Variable((nr, 1))
    Fu = cp.Variable((nr, nx))
    Ups1 = cp.Variable((m, nr))
    beta1 = cp.Variable((m, 1))
    eta1 = cp.Variable((n, 1), nonneg=True)
    eta2 = cp.Variable((nr, 1), nonneg=True)
    gam = cp.Variable(nonneg=True)
    eps = cp.Constant([[eps]])
    
    I = cp.Constant(np.eye(nx))
    Im = cp.Constant(np.eye(m))
    
    W0 = cp.Constant(NNo.NN['W0'])
    W1 = cp.Constant(NNo.NN['W1'])
    b0 = cp.Constant(NNo.NN['b0'])
    b1 = cp.Constant(NNo.NN['b1'])
    
    # Compose matrix inequality
    # 19/11: Spotted mistake in reduced_order_NN_weights - A[2][2] should be 2*Q1, NOT 2*Q2
    LMI = cp.bmat([ [gam*I, (-W0.T)@Q1 + Fu.T@J.T, -Fu.T + (W0.T)@M2.T, np.zeros((nx, 1)), np.zeros((nx, 1))],
                     [Q1.T@(-W0) + J@Fu, 2*Q1, -M1 - M2.T, -Q1@b0 + J@Fb + eta1, W1.T],
                     [-Fu + M2@W0, -M1.T - M2, 2*Q2, -Fb + M2@b0 + eta2, -Ups1.T],
                     [np.zeros((1, nx)), (b0.T)@(-Q1.T) + Fb.T@J.T + eta1.T, -Fb.T + (b0.T)@M2.T + eta2.T, eps, (b1.T - beta1).T],
                     [np.zeros((1, nx)), W1, -Ups1, b1.T - beta1, Im] ])
            
    constraints = [LMI >> 0]
    
    # Bound size of Q1 and Q2
    bound_q1 = cp.Constant(10**(5) * np.eye(n))  # 19/11: Can adjust to experiment with optimisation
    bound_q2 = cp.Constant(10**(5) * np.eye(nr))
    
    constraints += [Q1 <= bound_q1, Q2 <= bound_q2]
    
    # Define problem
    prob = cp.Problem(cp.Minimize(gam), constraints)
    prob.solve(solver='MOSEK')
    
    # Fetch optimised variables
    Q1v = Q1.value
    Q2v = Q2.value
    M2v = M2.value
    
    epsv = eps.value
    gamv = gam.value
    Ups1v = Ups1.value
    beta1v = beta1.value
    Fbv = Fb.value
    Fuv = Fu.value
    
    Ups0v = np.linalg.inv(Q2v) @ Fuv
    beta0v = np.linalg.inv(Q2v) @ Fbv
    
    NNr = Network()
    
    W0r = Ups0v
    b0r = beta0v
    W1r = Ups1v
    b1r = beta1v
    
    reduced_weights = [W0r, b0r, W1r, b1r]

    NNr.load(2, reduced_weights)
    
    return NNr, gamv, epsv[0][0], prob.status

def main():
    weights = np.array([np.array([-0.634700665707495, 1.656731026589558, 1.124487323633357, -0.795741403939511, -1.168109939313002, 0.869874974297699, -1.008855879224341, 1.914651700028183, 1.240317284056857, -1.420252553244148]).reshape(10, 1),
                   np.array([-0.258472107285207, -0.680982998070259, 1.560623889702786, -1.220971595926017, -0.027188941462718, -1.010335979063966, 0.835793863154799, 1.222569086520831, 0.259056187128039, 0.245551285836084]).reshape(10, 1),
                   np.array([0.082854200378300, 0.992706291627412, -1.256284879106125, 0.554918709010267, 0.317351602368993, 1.250828758360099, 1.679813903514902, -0.287094733070243, 1.701671362974561, -0.829936846466298]),
                   np.array([1.864581839050232])], dtype='object')

    NNo = Network()
    NNo.load(2, weights)

    eps = 5.0

    for i in range(9):
        NNr, gamv, epsv, status = L2_min_sector(NNo, i+1, eps)
        print(NNr.NN, gamv, epsv, status)

if __name__ == "__main__":
	main()	