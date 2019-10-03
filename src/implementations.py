'''
Return type: Note that all functions should return: (w, loss), 
which is the last weight vector of the method, 
and the corresponding loss value (cost function). 
Note that while in previous labs you might have kept track of all 
encountered w for iterative methods
, here we only want the last one.
'''

import numpy as np

from utils import helpers


# Generic gradient descent algorithm
def gradient_descent(y, tx, compute_loss, compute_gradient, initial_w, 
                     max_iters, gamma, batch_size=None, num_batch=None):
    
    shuffle = True
    
    if not batch_size:
        batch_size = len(y)
        num_batch = 1
        print('num_batch set to 1 as batch_size is None')
        shuffle = False
    
    if not num_batch:
        num_batch = 1
    
    
    losses = []
    ws = [initial_w]
    w = initial_w
    
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        
        gradient = 0
        for minibatch_y, minibatch_tx in helpers.batch_iter(y, tx, batch_size, num_batch, shuffle):
            gradient += compute_gradient(minibatch_y, minibatch_tx, w)
            
        gradient /= num_batch
        
        w = w - gamma * gradient
        
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws


def compute_loss_ls(y, tx, w):
    N = len(y)
    e = y - np.dot(tx,w)
    loss = 1/(2*N) * np.dot(e.T,e)
    return loss
    
def compute_gradient_ls(y, tx, w):
    N = len(y)
    e = y - np.dot(tx,w)
    return -1/N * np.dot(tx.T, e)


# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, compute_loss_ls, compute_gradient_ls, initial_w, 
                     max_iters, gamma)


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1
    num_batch = 1
    
    return gradient_descent(y, tx, compute_loss_ls, compute_gradient_ls, initial_w, 
                     max_iters, gamma, batch_size, num_batch)