import numpy as np

from utils import debugger 
from utils import misc
import implementations as impl
import cost

class Model:
    """A genereic framework for a trainable model

    E.g. if you set model_name = 'logistic_regression', then you can
    learn the weights simply by calling learn_weights(y, x), similarly,
    based on these trained weights, you can call predict(x), to get
    a prediction.

    Currently supports:

    - logistic_regression
    - reg_logistic_regression
    - least_squares_GD
    - ridge_regression
    - least_squares
    """

    def __init__(self, model_name, w=None, learning_param=None, debug=True):

        # Set weights
        self.w = w

        # Set debug object
        if debug:
            self.dbg = debugger.Debugger(['loss', 'w'])
        else:
            self.dbg = None

        """Depending on the chosen model, we choose the approriate output, 
        loss prediction, and learning functions.
        """
        if model_name == 'logistic_regression':
            self.model_output = misc.lr_output
            self.compute_loss = cost.compute_loss_ce
            self.predict_output = misc.map_prediction

            max_iters = learning_param['max_iters']
            gamma = learning_param['gamma']
            
            self.learn = lambda y, x, w, dbg: impl.logistic_regression(y, x, w, max_iters, gamma, dbg)

        if model_name == 'reg_logistic_regression':
            self.model_output = misc.lr_output
            self.compute_loss = cost.compute_loss_reg_ce
            self.predict_output = misc.map_prediction

            max_iters = learning_param['max_iters']
            gamma = learning_param['gamma']
            lambda_ = learning_param['lambda_']
            self.learn = lambda y, x, w, dbg: impl.reg_logistic_regression(y, x, lambda_, w, max_iters, gamma, dbg)


        if model_name == 'least_squares_GD':
            self.model_output = np.dot
            self.compute_loss = cost.compute_loss_ls
            self.predict_output = misc.predict_ls

            max_iters = learning_param['max_iters']
            gamma = learning_param['gamma']
            
            self.learn = lambda y, x, w, dbg: impl.least_squares_GD(y, x, w, max_iters, gamma, dbg)

        if model_name == 'ridge_regression':
            self.model_output = np.dot
            self.compute_loss = cost.compute_loss_ls
            self.predict_output = misc.predict_ls

            lambda_ = learning_param['lambda_']
            
            self.learn = lambda y, x, w, dbg: impl.ridge_regression(y, x, lambda_)

        if model_name == 'least_squares':
            self.model_output = np.dot
            self.compute_loss = cost.compute_loss_ls
            self.predict_output = misc.predict_ls

            self.learn = lambda y, x, w, dbg: impl.least_squares(y, x)

        
    def predict(self, x):
        """Predict y given x
        """
        return self.predict_output(self.model_output(x, self.w))

    def learn_weights(self, y, x):
        """Learn the weights of the model given the data y, x
        """
        print('learning weights...')
        self.w, _ = self.learn(y, x, self.w, self.dbg)
        print('done.')

    def loss(self, y, x):
        """Compute the loss given the data
        """
        self.compute_loss(y, x, self.w)

