import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    x_exp = x**2
    for i in range(3, degree+1):
        x_exp = np.concatenate((x_exp, x**i), axis=1)

    return x_exp


def augment_features(x, augment_param):
    degree = augment_param['degree']

    x_pol = build_poly(x, degree)

    x_aug = np.concatenate((x, x_pol), axis=1)

    return x_aug