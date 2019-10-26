import numpy as np


def build_poly(x, degree):
    n = len(x)
    vector = np.ones((1, n))
    matrix = vector
    
    for column in range(degree):
        vector = vector * x
        matrix = np.concatenate((matrix, vector), axis=0)
    
    return matrix.T
