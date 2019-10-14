'''
Return type: Note that all functions should return: (w, loss), 
which is the last weight vector of the method, 
and the corresponding loss value (cost function). 
Note that while in previous labs you might have kept track of all 
encountered w for iterative methods
, here we only want the last one.
'''

import numpy as np

from utils import misc
import cost


# Generic gradient descent algorithm
def gradient_descent(y, tx, compute_loss, compute_gradient, initial_w, 
                     max_iters, gamma, batch_size=None, num_batch=None, debugger=None):
    '''(Stochastic) Gradient descent algorithm'''
                    
    shuffle = True

    # If batch_size is none, apply standard GD
    if not batch_size:
        batch_size = len(y)
        num_batch = 1
        shuffle = False
    
    if not num_batch:
        num_batch = 1
    
    w = initial_w
    
    for _ in range(max_iters):
        loss = compute_loss(y, tx, w)
        
        gradient = 0
        for minibatch_y, minibatch_tx in misc.batch_iter(y, tx, batch_size, num_batch, shuffle):
            gradient += compute_gradient(minibatch_y, minibatch_tx, w)
            
        gradient /= num_batch
        
        w = w - gamma * gradient
        
        if debugger:
            debugger.add_item('loss', loss)
            debugger.add_item('w', w)
    
    return w, loss

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma, debugger=None):
    w, loss = gradient_descent(y, tx, cost.compute_loss_ls, cost.compute_gradient_ls, initial_w, 
                     max_iters, gamma, debugger=debugger)
    '''Gradient descent algorithm for least squares'''
    
    return w, loss

# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, debugger=None):
    '''Stochastic Gradient descent algorithm for least squares'''
    batch_size = 1
    num_batch = 1
    
    w, loss = gradient_descent(y, tx, cost.compute_loss_ls, cost.compute_gradient_ls, initial_w, 
                     max_iters, gamma, batch_size, num_batch, debugger=debugger)
    
    return w, loss

# Least squares regression using normal equations
def least_squares(y, tx):
    '''Solving least squares via the normal equations'''
    N = len(y)
    [weights, residuals, _, _] = np.linalg.lstsq(tx, y, rcond=None)
    loss = residuals / (2*N)
    
    return weights, loss

# Ridge regression using normal equations
def ridge_regression_old_version(y, tx, lambda_):
    '''Solving least squares via the normal equations'''
    N = len(y)

    xtx_inv = np.linalg.inv(tx.T @ tx + lambda_ * (2*N) * np.eye(tx.shape[1]))
    return (xtx_inv @ (tx.T)) @ y

def ridge_regression(y, tx, lambda_):
    '''Solving ridge regression via the normal equations'''
    N = len(y)
    M = tx.shape[1]
    
    a = tx.T @ tx + lambda_ * (2*N) * np.identity(M)
    b = tx.T @ y
    
    [weights, residuals, _, _] = np.linalg.lstsq(a, b, rcond=None)
    loss = residuals / (2*N)
    
    return weights, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, debugger=None):
    '''Logistic regression using gradient descent or GD'''
    return gradient_descent(y, tx, cost.compute_loss_ce, cost.compute_gradient_logreg, initial_w, 
                     max_iters, gamma, batch_size=None, num_batch=None, debugger=debugger)

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, debugger=None):
    '''Logistic regression using gradient descent or GD'''
    compute_gradient = lambda y, tx, w: cost.compute_gradient_reg_logreg(y, tx, w, lambda_)
    return gradient_descent(y, tx, cost.compute_loss_ce, compute_gradient, initial_w, 
                     max_iters, gamma, batch_size=None, num_batch=None, debugger=debugger)

