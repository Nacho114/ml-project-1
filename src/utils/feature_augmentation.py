import numpy as np
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree."""
    if(degree < 2):
        return np.array([])

    n = len(x)
    vector = x*x
    matrix = vector
    
    for column in range(2, degree):
        vector = vector * x
        matrix = np.concatenate((matrix, vector), axis=0)
    
    return matrix.T
    
def add_bias(x)
    return = np.ones((x.shape[0], 1))

def augment_features(x, augment_param):
    degree = augment_param['degree']

    x_pol = build_poly(x, degree)

    x_aug = np.concatenate((x, x_pol), axis=1)

    return x_aug
