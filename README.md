# ml-project-1
Dream Team project

## Description 
Project 1 of the "Machine Learning" course (CS433), given in the Fall semester of 2019.

## Instructions
To run our project, create a "Data" folder at the root of the directory and put the test.csv and train.csv files in it. Then navigate to "src" and run ```python run.py```.

## Code structure
Methods in **implementations.py** use different cost functions or gradient computation that can be found in **cost.py**. A big generic gradient descent method is implemented and used by all gradient descent models with different parameters.

Definition of **gradient_descent** method:
``
def gradient_descent(y, tx, compute_loss, compute_gradient, initial_w, 
                     max_iters=0, gamma=10e-6, batch_size=None, num_batch=None, debugger=None, dynamic_gamma=False):
``
