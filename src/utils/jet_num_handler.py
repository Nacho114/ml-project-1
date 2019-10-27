import numpy as np

from utils import constant

def get_undef_cols_idx(x, undef_val):
    """
    Find columns where all entries are undefined
    """
    undef_col_idx = []
    for col_idx in range(x.shape[1]):
        column = x[:, col_idx]
        if((column == undef_val).all()):
            undef_col_idx.append(col_idx)

    return undef_col_idx

def split_by_jet_num(x, y):
    """
    Create splot of data based on jet_num
    and also return the index in original data
    """
    y_not_empty = y is not None

    x_split = []
    y_split = []
    
    jet_num_to_idx = []

    # jet_num 0
    jet_num = 0
    mask = x[:, constant.JET_NUM_COL] == jet_num
    x_split.append(x[mask])
    if y_not_empty: 
        y_split.append(y[mask])

    jet_num_to_idx.append(mask)

    # jet_num 1
    jet_num = 1
    mask = x[:, constant.JET_NUM_COL] == jet_num
    x_split.append(x[mask])
    if y_not_empty: 
        y_split.append(y[mask])

    jet_num_to_idx.append(mask)

    # jet_num 2 and 3
    jet_nums = (2, 3)
    mask2 = x[:, constant.JET_NUM_COL] == jet_nums[0] 
    mask3 = x[:, constant.JET_NUM_COL] == jet_nums[1]

    mask = np.ma.mask_or(mask2, mask3)

    x_split.append(x[mask])
    if y_not_empty: 
        y_split.append(y[mask])

    jet_num_to_idx.append(mask)


    # merge jet_num_idx
    x_split_reduce = []
    x_split_reduce.append(x_split[0])
    x_split_reduce.append(x_split[1])
    x_split_reduce.append(x_split[2])
    
    y_split_reduce = []
    
    if y_not_empty:
        y_split_reduce.append(y_split[0])
        y_split_reduce.append(y_split[1])
        y_split_reduce.append(y_split[2])

    return x_split_reduce, y_split_reduce, jet_num_to_idx

def clean_split(x_split):
    """
    Given a data split x_split (by jet_num), we remove the undefined
    columns of the respective data.
    """
    clean_x_split = []

    undef_idx_of_jet_num = constant.UNDEF_IDX_OF_JET_NUM

    for idx, x in enumerate(x_split):
        undef_col_idx = undef_idx_of_jet_num[idx].copy()
        undef_col_idx.append(constant.JET_NUM_COL)
        new_x = np.delete(x, undef_col_idx, 1)
        clean_x_split.append(new_x)

    return clean_x_split