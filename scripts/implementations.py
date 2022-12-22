''' CS-433 Machine Learning: Project 1
This file contains methods for training models such as least squares regression, ridge regression, 
gradient descent, stochastic gradient descent,  logistic regression and regularized logistic regression. 
'''

import numpy as np
from load_data import *

########## PREDICTION ##################

def prediction(w, x): 
    ''' Given weights w and data x, computes the predicted labels y_pred'''
    y_pred = x @ w
    y_pred = np.where(y_pred > 0, 1, -1)
    return y_pred

def evaluate_accuracy(y_pred, y):
    ''' Computes the accuracy of prediction y_pred with regards to the true labels y'''
    equals = np.where(y_pred == y, 1, 0) 
    accuracy = np.sum(equals) / y.shape[0]
    return round(accuracy, 4)


############## MODELS ###############


def compute_loss(y, x, w):
    """Calculate the MSE loss
    Args:
        y: numpy array of shape=(N, )
        x: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return 1/(2 * x.shape[0]) * np.sum((y-x@w)**2, axis=0)

def compute_gradient(y, x, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        x: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (dim, ) (same shape as w), containing the gradient of the loss at w.
    """
    return -(1/x.shape[0]) * (x.T @ (y-x @ w))


def batch_iter(y, x, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'x')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `x`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_x in batch_iter(y, x, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_x = x[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_x = x
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_x[start_index:end_index]

    
def mean_squared_error_gd(y, tx, max_iters, gamma):
    """Linear regression using gradient descent
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE scalar.
    """
    w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma * grad
    loss = compute_loss(y,tx,w)
    return w, loss


def mean_squared_error_sgd(y, tx, batch_size, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE scalar.
    """
    w = np.zeros(tx.shape[1])
    
    for n_iter in range(max_iters):

        for minibatch_y, minibatch_x in batch_iter(y, tx, batch_size):
            grad = compute_gradient(minibatch_y, minibatch_x, w)
            w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE scalar.
    """
    w = np.linalg.inv(tx.T @ tx)@tx.T@y
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        x: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE scalar.
    """
    N, d = tx.shape
    w = np.linalg.inv(tx.T @ tx + (2 * N * lambda_) * np.identity(d)) @ tx.T @ y
    loss = compute_loss(y,tx,w)
    return w, loss

def sigmoid(x):
    return 1. /(1. + np.exp(-x))

def calculate_loss(y, x, w):
    """ Compute the cost by negative log likelihood.
    Args:
        y:  shape=(N, 1)
        x: shape=(N, D)
        w:  shape=(D, 1) 
    Returns:
        a non-negative loss
    """
    assert y.shape[0] == x.shape[0]
    assert x.shape[1] == w.shape[0]

    return 1/len(y)*(np.log(1+np.exp(x@w))-y*(x@w)).sum()

def calculate_gradient(y, x, w):
    """ Computes the loss gradient.
    Args:
        y:  shape=(N, 1)
        x: shape=(N, D)
        w:  shape=(D, 1) 
    Returns:
        a vector of shape (D, 1)
    """
    return (1/x.shape[0]) * (x.T @ (sigmoid(x@w) -y))

def logistic_regression(y, tx, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])
    for iter in range(max_iters):
        # Get loss and update weights
        grad = calculate_gradient(y,tx,w)
        loss = calculate_loss(y,tx,w)
        w = w -gamma*grad
        # Convergence criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


def calculate_regularized_loss(y, x, w, lambda_):
    """ Return the loss and gradient.
    Args:
        y:  shape=(N, 1)
        x: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar
    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    grad = calculate_gradient(y,x,w)+ 2 * lambda_ * w
    loss = calculate_loss(y,x,w)+lambda_*np.linalg.norm(w)**2
    return loss, grad


def reg_logistic_regression(y, tx,lambda_, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])
    for iter in range(max_iters):
        # Get loss and update weights
        loss, grad = calculate_regularized_loss(y,tx,w,lambda_)
        w = w-gamma*grad

        # Convergence criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w,losses[-1]
