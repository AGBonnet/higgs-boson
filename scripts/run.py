''' CS-433 Machine Learning: Project 1
This file is used to produce the final test labels, once we have selected our optimal model. 
'''

import numpy as np
from implementations import *
from load_data import *
from train import *

def run(model_name, preHyp, modelHyp):
    print('Running model', model_name.__name__, 'with\n\tPre-processing hyperparameters:', preHyp,'\n\tModel hyperparameters:', modelHyp) 
    # Load data using helper functions
    subsample=False
    train_data, train_labels, train_ids, test_data, test_ids = load_data(subsample)

    preHypFull = preHyp
    preHypFull['beta'] = None
    x_train, y_train = pre_process(x = train_data, y = train_labels, **preHyp)
    x_train_full, _ = pre_process(x = train_data, y = None, **preHypFull)
    x_test, _ = pre_process(x = test_data, y = None, **preHypFull)

    # Train optimal model with optimal hyperparameters on training set
    w, _ = model_name(y=y_train, tx=x_train, **modelHyp)

    # Produce predicted labels for training set
    trainPred = prediction(w, x_train_full)
    trainAcc = evaluate_accuracy(trainPred, train_labels)
    print('Training accuracy:', trainAcc)
    
    # Produce predicted labels for test set
    testPred = prediction(w, x_test)

    # Store labels in .csv file
    output_name = '../data/test_labels.csv'
    create_csv_submission(test_ids, testPred, output_name)

if __name__ == '__main__':
    # Pre-process training and test data with optimal hyperparameters (do not remove outliers for test)
    optModelName = ridge_regression
    optPreHyp = {'degree': 6, 'alpha': 1, 'beta': None}
    optModelHyp = {'lambda_': 1e-06}

    run(optModelName, optPreHyp, optModelHyp)