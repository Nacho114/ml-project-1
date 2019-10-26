import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    x_exp = x
    for j in range(2, degree+1):
        x_exp = np.concatenate((x_exp, x**j), axis=1)
    return x_exp