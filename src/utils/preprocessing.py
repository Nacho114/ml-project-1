import numpy as np

from utils import jet_num_handler

def normalise(x):
    """
    Normalise 
    """

    if np.all(x == x[0]):
        return x

    return (x - x.mean(axis=0)) / x.std(axis=0)

def most_frequent(x, extended_output=False):
    """
    Get the most frequent value in an array
    """
    counter = {}
    max_val = x[0]
    counter[max_val] = 1
    for val in x[1:]:
        if(val in counter):
            counter[val] += 1
            if counter[val] > counter[max_val]:
                max_val = val
        else:
            counter[val] = 1
                
    return (max_val, counter[max_val]) if extended_output else max_val

def replace(x, err_val, find_replacement):
    """
    Replace each occurence of a specified value in an array
    according to a specified replacement function
    """
    replacement = find_replacement(x[x != err_val])

    replaced = x.copy()
    replaced[replaced == err_val] = replacement
    
    return replaced
   
def clean_data(x, err_val, find_replacement):
    """
    Clean a matrix by replacing errors values in each column
    according to a specified replacement function
    """
    x_clean = np.zeros(x.shape)
    nb_features = x.shape[1]
    
    for feature in range(nb_features):
        x_clean[:, feature] = replace(x[:, feature], err_val, find_replacement)
        
    return x_clean


def preprocess(x, to_replace, do_normalise=True, add_bias=True):
    """
    Preprocess the data matrix

    1. to_replace clean a matrix by replacing errors values in each column
    according to a specified replacement function.

    e.g. [(-111, 'mean')] will replace all occurances of -111 with the mean value
    of that featrue value over all samples (excluding -111).

    2. do_normalise normalises the data if set true

    3. add_bias adds a column of ones for the bias term
    """

    replace_method_map = {
        'mean': np.mean,
        'most_frequent': most_frequent,
        'median': np.median
    }

    # for each err_val to be replaced (to_replace), replace
    # with corresponding replace method
    for err_val, replace_method in to_replace:
        find_replacement = replace_method_map[replace_method]
        x = clean_data(x, err_val, find_replacement)

    if do_normalise:
        x = normalise(x)

    # Add bias column of 1s
    if add_bias:
        nb_samples = nb_samples = x.shape[0]
        first_col = np.ones((nb_samples, 1))
        x = np.concatenate((first_col, x), axis=1)
    
    return x


def preprocess_jet_num(x, y, to_replace, do_normalise=True, add_bias=True):

    x_split, y_split = jet_num_handler.split_by_jet_num(x, y)
    clean_x_split = jet_num_handler.clean_split(x_split)

    cleaner_x_split = []

    for x_ in clean_x_split:
        cleaner_x = preprocess(x_, to_replace, do_normalise, add_bias)
        cleaner_x_split.append(cleaner_x)

    return cleaner_x_split, y_split

