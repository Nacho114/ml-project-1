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

    return x_split, y_split

def clean_split(x_split):
    clean_x_split = []

    undef_idx_of_jet_num = constant.UNDEF_IDX_OF_JET_NUM

    for idx, x in enumerate(x_split):
        undef_col_idx = undef_idx_of_jet_num[idx]
        new_x = np.delete(x, undef_col_idx, 1)
        clean_x_split.append(new_x)

    return clean_x_split

