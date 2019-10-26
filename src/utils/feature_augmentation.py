import numpy as np


def build_poly(x, degree):
    if(degree < 2):
        return np.array([])

    n = len(x)
    vector = x*x
    matrix = vector
    
    for column in range(2, degree):
        vector = vector * x
        matrix = np.concatenate((matrix, vector), axis=0)
    
    return matrix.T
