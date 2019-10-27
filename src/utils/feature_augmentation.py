import numpy as np
    
def build_poly(x, degrees):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""

    x_exp = x**degrees[0]
    for degree in degrees[1:]:
        x_exp = np.concatenate((x_exp, x**degree), axis=1)

    return x_exp
    
def get_bias(x):
    """Return the bis term (a vector of ones)"""
    return np.ones((x.shape[0], 1))
    
def cross_features(x):
    """Return a matrix with all the cross products of the data (correlations)"""
    cross = np.zeros((x.shape[0], 1))
    
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            cross = np.concatenate((cross, (x[:,i]*x[:,j])[:, np.newaxis]), axis=1)
            
    return cross[:, 1:]

def augment_features(x, augment_param):
    """augment x with the given parameters given in augment data"""
    required_params = ['degrees', 'add_bias', 'add_cross', 
                        'add_tanh', 'cumulative']

    for param in required_params:
        if param not in augment_param:
            augment_param[param] = None

    degrees = augment_param['degrees']
    add_bias = augment_param['add_bias']
    add_cross = augment_param['add_cross']
    add_tanh = augment_param['add_tanh']
    is_cumulative = augment_param['cumulative']

    x_aug = x

    if degrees:
        x_pol = build_poly(x, degrees)
        x_aug = np.concatenate((x_aug, x_pol), axis=1)

    if add_cross:
        x_arg = x_aug if is_cumulative else x
        x_cross = cross_features(x_arg)
        x_aug = np.concatenate((x_aug, x_cross), axis=1)   
        
    if add_tanh:
        x_arg = x_aug if is_cumulative else x
        tanh = np.tanh(x_arg)
        x_aug = np.concatenate((x_aug, tanh), axis=1) 

    if add_bias:
        bias = get_bias(x)
        x_aug = np.concatenate((bias, x_aug), axis=1)

    return x_aug