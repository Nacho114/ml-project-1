# ml-project-1
Dream Team project

## Authors

- Thierry Bossy
- Raphael Strebel
- Ignacio S. Aleman

## Description 
Project 1 of the "Machine Learning" course (CS433), given in the Fall semester of 2019.

## Instructions

### Data

To run our project, create a folder named "data" at the root of the project directory and put the test.csv and train.csv files in it.

### Execution

Navigate to "src" and run ```python run.py```. (Note that you need at least Python 3.6 to run the code)

## Code structure

Methods in **implementations.py** use different cost functions or gradient computation that can be found in **cost.py**. A big generic gradient descent method is implemented and used by all gradient descent models with different parameters.

Definition of **gradient_descent** method:
``
def gradient_descent(y, tx, compute_loss, compute_gradient, initial_w, 
                     max_iters=0, gamma=10e-6, batch_size=None, num_batch=None, debugger=None, dynamic_gamma=False):
``
