''' CS-433 Machine Learning: Project 1
This file contains methods for loading the data and pre-processing it, including missing value replacement,
removing outliers, polynomial expansion and standardization/normalization.  
'''

import numpy as np
from helpers import *

############# LOADING DATA ###################

def load_data(sub_sample=False):
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'

    train_labels, train_data, train_ids = load_csv_data(train_path, sub_sample=sub_sample)
    _, test_data, test_ids = load_csv_data(test_path, sub_sample=sub_sample)
    
    return train_data, train_labels, train_ids, test_data, test_ids

############# PRE-PROCESSING DATA ##############

def replace_missing(data, alpha):
    '''
    Removes all columns with more than threshold % missing values. 
    Replace the remaining missing values by their respective median.
    '''
    # Remove columns with more than threshold % missing values
    num_missing = (data == -999).sum(axis=0)
    to_replace = (num_missing <= alpha * data.shape[0])
    new_data = data[:, to_replace]

    # Replace missing values with column-wise median
    missing = (new_data == -999)
    for i in range(new_data.shape[1]):
        med = np.median(new_data[~missing[:,i],i])
        new_data[missing[:,i],i] = med

    return new_data

def remove_outliers(data, beta):
    ''' 
    Removes all rows of the data matrix which contain features that are beta standard deviations from the mean
    '''
    inliers = np.full(data.shape[0], True)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    for i in range(data.shape[1]):
        inliers *= (abs(data[:,i]-means[i]) <= beta * stds[i]) # * same as logical AND
    return data[inliers, :], inliers


def polynomial_expansion(data, degree): 
    '''
    Expands each feature of data matrix to the given degree, with an added column of 1's.
    '''
    n, m = data.shape
    poly = np.ones((n, m*degree+1))
    for i in range(m):
        for deg in range(1, degree+1):
            col = i*degree + deg 
            poly[:,col] = data[:,i] ** deg
    return poly


def standardize(data):
    ''' Standardize each column of the array data to follow a N(0,1) distribution'''
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    idx = np.nonzero(stds)
    std_data = np.ones(data.shape)
    std_data[:, idx] = (data[:, idx] - means[idx]) / stds[idx]
    return std_data

def normalize(data):
    ''' Rescale each column of the array data to the [0,1] interval'''
    maxs = np.max(data, axis=0)
    mins = np.min(data, axis=0)
    idx = np.nonzero(maxs-mins)
    norm_data = np.ones(data.shape)
    norm_data[:, idx] = (data[:, idx] - mins[idx]) / (maxs[idx] - mins[idx])
    return norm_data

def pre_process(x, y = None, alpha = None, beta = None, degree = None):
    '''
    Pre-process data matrix x by replacing missing values, removing outliers, polynomial expansion and standardization.
    alpha: missing values threshold
    beta: outliers threshold
    degree: polynomial expansion maximum degree
    '''
    if alpha:
        x = replace_missing(x, alpha)
    if beta:
        x, inliers = remove_outliers(x, beta)
        if y is not None:
            y = y[inliers]
    if degree: 
        x = polynomial_expansion(x, degree)
    x = standardize(x)
    return x, y

