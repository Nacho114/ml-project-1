import numpy as np
    
def build_poly(x, degrees):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""

    x_exp = x**degrees[0]
    for degree in degrees[1:]:
        x_exp = np.concatenate((x_exp, x**degree), axis=1)

    return x_exp
    
def add_bias(x):
    return np.ones((x.shape[0], 1))
    
def cross_features(x):
    cross = np.zeros((x.shape[0], 1))
    
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            cross = np.concatenate((cross, (x[:,i]*x[:,j])[:, np.newaxis]), axis=1)
            
    return cross[:, 1:]
def augment_features(x, augment_param):
    degree = augment_param['degree']

    x_pol = build_poly(x, degree)

    x_aug = np.concatenate((x, x_pol), axis=1)

    return x_aug
