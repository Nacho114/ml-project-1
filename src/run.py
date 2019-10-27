import numpy as np

from utils import loader
from utils import preprocessing as pp 
from utils import constant
import model
from utils import misc

np.random.seed(114)


def get_models(model_type, learning_param, nb_models=3):
    """generate models based on params"""
    models = []

    for _ in range(nb_models):
        curr_model = model.Model(model_type, learning_param=learning_param)
        models.append(curr_model)

    return models


def train_models(train_data, models):
    """Train models with the given data"""
    x_split = train_data['x_split']
    y_split = train_data['y_split']

    for idx, x_tr in enumerate(x_split):

        y_tr = y_split[idx]

        curr_model = models[idx]
        curr_model.learn_weights(y_tr, x_tr)


def get_predictions(test_data, models):
    """predict output based on model and data"""
    x_split = test_data['x_split']

    predictions = []
    for idx, x_partition in enumerate(x_split):

        curr_model = models[idx]
        y_ = curr_model.predict(x_partition)
        predictions.append(y_)

    jet_num_to_idx = test_data['jet_num_to_idx']
    predictions = misc.merge_predictions(predictions, jet_num_to_idx)

    return predictions


def main():

    ### LOAD TEST DATA ###

    DATA_TEST_PATH = '../data/test.csv' 
    DATA_TRAIN_PATH = '../data/train.csv' 

    data = {}

    y, x, ids = loader.load_csv_data(DATA_TEST_PATH)

    data['test'] = {
        'y': y,
        'x': x,
        'ids': ids
    }

    y, x, ids = loader.load_csv_data(DATA_TRAIN_PATH)

    data['train'] = {
        'y': y,
        'x': x,
        'ids': ids
    }

    # CLEAN DATA OPTION

    to_replace = [(constant.UNDEF_VAL, 'mean')]

    do_normalise = True

    # FEATURE AUGMENTATION PARAMS

    augment_param_0 = { 
        'degrees': [2,3], 
        'add_bias' : True, 
        'add_cross': True, 
        'add_tanh': True, 
        'cumulative': False 
    }

    augment_param_1 = {
        'degrees': [2],
        'add_bias' : True,
        'add_cross': True,
        'add_tanh': True,
        'cumulative': True
    }

    augment_param_23 = {
        'degrees': [2],
        'add_bias' : True,
        'add_cross': True,
        'add_tanh': True,
        'cumulative': True
    }

    augment_param_list = [augment_param_0, augment_param_1, augment_param_23]


    # Split test based on jet_num 
    x_split, y_split, jet_num_to_idx = \
            pp.preprocess_jet_num(x=data['test']['x'], 
                                  y=data['test']['y'], 
                                  to_replace=to_replace, 
                                  do_normalise=do_normalise, 
                                  augment_param_list=augment_param_list)

    data['test']['x_split'] = x_split
    data['test']['y_split'] = y_split
    data['test']['jet_num_to_idx'] = jet_num_to_idx


    # Split train based on jet_num 
    x_split, y_split, jet_num_to_idx = \
            pp.preprocess_jet_num(x=data['train']['x'], 
                                  y=data['train']['y'], 
                                  to_replace=to_replace, 
                                  do_normalise=do_normalise, 
                                  augment_param_list=augment_param_list)

    data['train']['x_split'] = x_split
    data['train']['y_split'] = y_split
    data['train']['jet_num_to_idx'] = jet_num_to_idx


    # Model parameters

    model_type = 'ridge_regression'
    learning_param = { 'lambda_': 1e-6 }

    # get models
    models = get_models(model_type, learning_param)

    # train models
    train_models(data['train'], models)

    # predict labels of test
    predictions = get_predictions(data['test'], models)


    ids = data['test']['ids']

    loader.create_csv_submission(ids, predictions, "final_submission.csv")

if __name__ == '__main__':
    main()
