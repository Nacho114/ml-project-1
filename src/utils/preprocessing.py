import numpy as np

def normalise(x):
    """
    Normalise x
    """
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

def most_frequent_new(x):
    counts = np.bincount(x)
    return np.argmax(counts)

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
