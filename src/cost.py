import numpy as np


def cross_entropy(x):
    pass


## USE CROSS ENTROPY LOSS

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



### Loss functions

def compute_loss_ls(y, tx, w):
    N = len(y)
    e = y - tx @ w
    loss = 1/(2*N) * e.T @ e
    
    return loss


def compute_loss_ce(y, tx, w):
    N = len(y)
    Z = tx @ w
    y_ = sigmoid(Z)
    pos_mask = y > 0
    loss = -(1/N) * (np.sum(np.log(y_[pos_mask])) + np.sum(np.log(1 - y_[~pos_mask])))
    
    return loss


### Gradient of loss functions
    
def compute_gradient_ls(y, tx, w):
    N = len(y)
    e = y - tx @ w
    gradient = -(1/N) * tx.T @ e
    
    return gradient


def compute_gradient_logreg(y, tx, w):
    '''' Gradient of logistic regression with 
        cross entropy loss
    '''
    N = len(y)
    M = sigmoid(tx @ w)
    gradient = (1/N) * tx @ M
    
    return gradient

