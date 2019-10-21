import numpy as np


### Function def
def sigmoid(x):
    """Sigmoid function"""
    
    return 1 / (1 + np.exp(-x))

def stable_log(x):
    eps = 1e-15
    return np.log(x + eps)


### Loss functions
def compute_loss_ls(y, tx, w):
    """Computes the loss of the least squares"""
    
    N = len(y)
    e = y - tx @ w
    loss = 1/(2*N) * e.T @ e
    
    return loss

def compute_loss_ce(y, tx, w):
    """Computes the cross entropy loss for the sigmoid function"""
    
    eps = 1e-15
    N = len(y)
    Z = tx @ w
    y_ = sigmoid(Z)
    pos_mask = y > 0

    loss = -(1/N) * (stable_log(y_[pos_mask]).sum() + stable_log(1 - y_[~pos_mask]).sum())
    
    return loss

def compute_loss_reg_ce(y, tx, w, lambda_):
    return compute_loss_ce(y, tx, w)  + lambda_/2 * np.linalg.norm(w)

### Gradient of loss functions
def compute_gradient_ls(y, tx, w):
    '''Gradient of least squares'''
    
    N = len(y)
    e = y - tx @ w
    gradient = -(1/N) * tx.T @ e
    
    return gradient

def compute_gradient_logreg(y, tx, w):
    '''Gradient of logistic regression with cross entropy loss'''
    
    N = len(y)
    e = sigmoid(tx @ w) - y
    gradient = tx.T @ e
    
    return gradient

def compute_gradient_reg_logreg(y, tx, w, lambda_):
    '''Gradient of regularised logistic regression with cross entropy loss'''
    
    usual_grad = compute_gradient_logreg(y, tx, w)
    
    return usual_grad + lambda_*w