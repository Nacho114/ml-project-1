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
    
    shuffle = True
    
    if not batch_size:
        batch_size = len(y)
        num_batch = 1
        print('num_batch set to 1 as batch_size is None')
        shuffle = False
    
    if not num_batch:
        num_batch = 1
    
    w = initial_w
    
    for n_iter in range(max_iters):
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


def compute_loss_ls(y, tx, w):
    N = len(y)
    e = y - np.dot(tx,w)
    loss = 1/(2*N) * np.dot(e.T,e)
    
    return loss
    
def compute_gradient_ls(y, tx, w):
    N = len(y)
    e = y - np.dot(tx,w)
    gradient = -1/N * np.dot(tx.T, e)
    
    return gradient


# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma, debugger=None):
    w, loss = gradient_descent(y, tx, compute_loss_ls, compute_gradient_ls, initial_w, 
                     max_iters, gamma, debugger=debugger)
    
    return w, loss


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, debugger=None):
    batch_size = 1
    num_batch = 1
    
    gradient_descent(y, tx, compute_loss_ls, compute_gradient_ls, initial_w, 
                     max_iters, gamma, batch_size, num_batch, debugger=debugger)
    
    return w, loss



# Least squares regression using normal equations
def least_squares(y, tx):
    N = len(y)
    [weights, _, _, _] = np.linalg.lstsq(tx, y, rcond=None)
    
    return weights

# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):

    xtx_inv = np.linalg.inv(tx.T @ tx + lambda_ * np.eye(tx.shape[1]))
    return (xtx_inv @ (tx.T)) @ y

def ridge_regression_thierry_version(y, tx, lambda_):
    N = len(y)
    M = tx.shape[1]
    
    a = tx.T @ tx + lambda_*2*N*np.identity(M)
    b = tx.T @ y
    
    [weights, residuals, rank, s] = np.linalg.lstsq(a, b, rcond=None)
    mse = residuals / (2*N)
    
    return weights
    

# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma, debugger=None):
    gradient_descent(y, tx, cost.compute_loss_ce, cost.compute_gradient_logreg, initial_w, 
                     max_iters, gamma, batch_size=None, num_batch=None, debugger=debugger)

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass

