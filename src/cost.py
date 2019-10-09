import numpy as np


def cross_entropy(x):
    pass


## USE CROSS ENTROPY LOSS

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

### Loss functions

def compute_loss_ls(y, tx, w, cost=mse):
    N = len(tx)
    e = y - np.dot(tx,w)
    return  1/(2*N) * np.dot(e,e)


### Gradient of loss functions
    
def compute_gradient_ls(y, tx, w):
    N = len(y)
    e = y - np.dot(tx,w)
    return -1/N * np.dot(tx.T, e)


def compute_gradient_lr():
    pass