import numpy as np

def find_jet_num_column(x):
    """ this works as long as only one column has values between 0 and 3. Otherwise we have to check that every cell is an integer, but it works on this data  """
    for col_idx in range(x.shape[1]):
        column = x[:,col_idx]
        if(0 <= np.min(column) and np.max(column) <= 3):
            return col_idx
        
def get_undef_cols_idx(m, undef_val=-999):
    undef_col_idx = []
    for col_idx in range(m.shape[1]):
        column = m[:,col_idx]
        if((column == undef_val).all()):
            undef_col_idx.append(col_idx)
    return undef_col_idx

def get_x_jet_num_reduced(x_jet_num, undef_val=-999):
    undef_col_idx = get_undef_cols_idx(x_jet_num)
    return np.delete(x_jet_num, undef_col_idx, 1)