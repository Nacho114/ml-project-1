import numpy as np

from utils import loader
from utils import preprocessing as pp 
from utils import constant
import model
from utils import misc

def main():

    ### LOAD TEST DATA ###

    DATA_TEST_PATH = '../data/test.csv' 
    y, x, ids = loader.load_csv_data(DATA_TEST_PATH)
    y = None
    nb_samples = len(x)


    # CLEAN AND AUGMENT DATA
    to_replace = [(constant.UNDEF_VAL, 'most_frequent')]

    augment_param = {
        'degrees': [2],
        'add_bias' : True,
        'add_cross': True,
        'add_tanh': True
    }

    # Split based on jet_num 
    x_split, y_split, jet_num_to_idx = pp.preprocess_jet_num(x=x, y=y, to_replace=to_replace, 
                    do_normalise=True, augment_param=augment_param)


    # Model parameters
    default_model = 'ridge_regression'
    learning_param = { 'lambda_': 1e-6 }

    # Load trained weights
    file_name = 'weights.npy'
    weights = np.load(file_name) 


    acc_list = []

    predictions = []
    for idx, x_partition in enumerate(x_split):
        print(x_partition.shape) 
        model_ls = model.Model(default_model, weights[idx], learning_param)
        y_ = model_ls.predict(x_partition)
        predictions.append(y_)


    predictions = misc.merge_predictions(predictions, jet_num_to_idx)

    loader.create_csv_submission(ids, predictions, "final_submission.csv")

if __name__ == '__main__':
    main()
