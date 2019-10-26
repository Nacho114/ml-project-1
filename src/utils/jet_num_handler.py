import numpy as np

from utils import constant

def get_undef_cols_idx(x, undef_val):
    undef_col_idx = []
    for col_idx in range(x.shape[1]):
        column = x[:, col_idx]
        if((column == undef_val).all()):
            undef_col_idx.append(col_idx)

    return undef_col_idx

def split_by_jet_num(x, y):
    x_split = []
    y_split = []

    for jet_num in constant.JET_NUM_VAL:
        mask = x[:, constant.JET_NUM_COL] == jet_num
        x_split.append(x[mask])
        if not y is None:
            y_split.append(y[mask])
            
    x_split_23 = np.concatenate((x_split[2], x_split[3]), axis=0)
            
    x_split_reduce = []
    x_split_reduce.append(x_split[0])
    x_split_reduce.append(x_split[1])
    x_split_reduce.append(x_split_23)
    
    y_split_reduce = []
    
    if not y is None:
        y_split_23 = np.concatenate((y_split[2], y_split[3]), axis=0)
        
        y_split_reduce.append(y_split[0])
        y_split_reduce.append(y_split[1])
        y_split_reduce.append(y_split_23)

    return x_split_reduce, y_split_reduce

def clean_split(x_split):
    clean_x_split = []

    undef_idx_of_jet_num = constant.UNDEF_IDX_OF_JET_NUM

    for idx, x in enumerate(x_split):
        undef_col_idx = undef_idx_of_jet_num[idx].copy()
        undef_col_idx.append(constant.JET_NUM_COL)
        new_x = np.delete(x, undef_col_idx, 1)
        clean_x_split.append(new_x)

    return clean_x_split

