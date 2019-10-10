import numpy as np


def cross_entropy(x):
    pass


## USE CROSS ENTROPY LOSS

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

### Loss functions

def compute_loss_ls(y, tx, w):
    N = len(tx)
    e = y - np.dot(tx,w)
    return  1/(2*N) * np.dot(e,e)


def compute_loss_ce(y, tx, w):
    N = len(y)
    Z = tx @ w
    y_ = sigmoid(Z)
    pos_mask = y > 0
    return -(1/N) * ( np.sum(np.log(y_[pos_mask])) + np.sum(np.log(1 - y_[~pos_mask])) )


### Gradient of loss functions
    
def compute_gradient_ls(y, tx, w):
    N = len(y)
    e = y - np.dot(tx,w)
    return -1/N * np.dot(tx.T, e)


def compute_gradient_logreg(y, tx, w):
    '''' Gradient of logistic regression with 
        cross entropy loss
    '''
    N = len(y)
    M = sigmoid(np.dot(tx, w))
    return 1/N * tx @ M

