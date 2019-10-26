import numpy as np
import cost


def merge_predictions(y_pred_split, jet_num_to_idx):
    """ 
    Merge back predictions based on original index
    of the data
    """
    nb_samples = sum([len(y) for y in y_pred_split])
    pred = np.zeros(nb_samples)

    for y, mask in zip(y_pred_split, jet_num_to_idx):
        pred[mask] = y

    return pred


def lr_output(x, w):
    """
    Returns the binary predictions of logistic regression
    """
    z = cost.sigmoid(x @ w)
    return z


def map_prediction(y):
    """
    Given a vector of proababilities, returns the maximum apostetiori
    prediction (MAP), i.e. 1 <-> pr(y_i) > .5, and -1 otherwise
    """
    pred = np.ones(len(y)) 
    to_minus = y < .5 
    pred[to_minus] = -1
    return pred


def predict_ls(y_pred):
    """Generates class predictions given output of a linear model (or least squares [ls])"""
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

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

def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)

def random_data_split(y, x, k_fold):
    """Returns a random split of the data (test, train), s.t.
    size train = k_fold * size test
    """
    k_indices = build_k_indices(y, k_fold)
    k = 0
    return split_data(y, x, k_indices, k)


def split_data(y, x, k_indices, k):
    """"Splits the data y, x given k_indices and k"""

    x_tr = x[np.concatenate((k_indices[:k], k_indices[k+1:]))].reshape((-1, x.shape[1]))
    y_tr = y[np.concatenate((k_indices[:k], k_indices[k+1:]))].flatten()

    train_data = x_tr, y_tr
    
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]

    test_data = x_te, y_te

    return train_data, test_data



def single_cross_validation(y, x, k_indices, k, get_weights, compute_loss):
    """Return train and test loss of one cross validation"""
    
    x_tr = x[np.concatenate((k_indices[:k], k_indices[k+1:]))].reshape((-1, x.shape[1]))
    y_tr = y[np.concatenate((k_indices[:k], k_indices[k+1:]))].flatten()
    
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]

    # get loss of train data and optimal weights
    w, loss_tr = get_weights(y_tr, x_tr)

    # get loss of test data
    loss_te = compute_loss(y_te, x_te, w)

    return loss_tr, loss_te, w

def cross_validation(x, y, k_indices, get_weights, compute_loss):

    # split data in k fold
    k_fold = len(k_indices)
    
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    
    for k in range(k_fold):
        loss_tr, loss_te, _ = single_cross_validation(y, x, k_indices, k, get_weights, compute_loss)
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)

    mean_loss_tr = np.mean(losses_tr)
    mean_loss_te = np.mean(losses_te)
        
    return mean_loss_tr, mean_loss_te

