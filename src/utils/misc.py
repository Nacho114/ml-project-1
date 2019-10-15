import numpy as np
import cost

def predict(x, w):
    """
    Returns the binary predictions of logistic regression
    """
    z = cost.sigmoid(x @ w)
    return get_predictions(z)


def get_predictions(y):
    """
    Given a vector of proababilities, returns the optimal bayes
    prediction, i.e. 1 <-> pr(y_i) > .5, and 0 otherwise
    """
    pred = np.ones(len(y)) 
    to_minus = y < .5 
    pred[to_minus] = -1
    return pred

def accuracy(y, y_):
    """
    Returns accuracy of prediction y_ vis-a-vis truth y
    """
    return (y == y_).sum() / len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


### Cross validation

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    np.random.seed(seed)
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    x_tr = x[np.concatenate((indices[:k], indices[k+1:]))].flatten()
    y_tr = y[np.concatenate((indices[:k], indices[k+1:]))].flatten()
    
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    
    # ...
    
    return

def use_cross_validation(seed, x, y, degree, k_fold, lambda_):

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    
    for k in range(k_fold):
        loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)

    rmse_tr = np.mean(losses_tr)
    rmse_te = np.mean(losses_te)
        
    return