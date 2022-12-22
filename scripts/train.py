''' CS-433 Machine Learning: Project 1
This file contains methods for model training including K-fold cross-validation and hyperparameter tuning. 
'''

import numpy as np
import math

from load_data import *
from implementations import *


########## MODEL SELECTION #############

def cross_validation_split(x, K):
    # Create list of indices for cross validation
    N = x.shape[0] # Number of datapoints
    size = math.floor(N/K)  # Size of each portion
    idx = np.arange(N)      # Indices {0,1,...,N-1}
    idx = np.random.permutation(idx)  # Shuffled indices 
    validationIDX = [list(idx[i:i+size]) for i in range(0,size*K,size)]
    trainingIDX = [list(np.delete(idx,np.array(range(i,i+size)))) for i in range(0,size*K,size)]
    return trainingIDX, validationIDX


def kfoldCV(x, y, K, model_name, hyperparameter, beta=None):
    '''
    K-fold Cross validation: iterates through each train/validation set 
    and returns the average result of each fold.
    '''
    # For each cross-validation split, train model, predict labels, report accuracy
    trainingIDX, validationIDX = cross_validation_split(x, K)
    validAccuracies = []
    trainAccuracies = []

    for trainIDX, validIDX in zip(trainingIDX, validationIDX):
        # Split dataset using training/validation indices
        x_train = x[trainIDX,:]
        y_train = y[trainIDX]
        x_valid = x[validIDX,:]
        y_valid = y[validIDX]

        # If beta is not None, remove outliers for training and keep outliers for validation
        if beta:
            x_train, inliers = remove_outliers(x_train, beta)
            y_train = y_train[inliers]

        # Train arbitrary model using training set
        w_train, _ = model_name(y=y_train, tx=x_train, **hyperparameter) 

        # Predict labels using validation set, record validation accuracy
        valid_pred = prediction(w_train, x_valid)
        validAcc = evaluate_accuracy(y_valid, valid_pred)
        validAccuracies.append(validAcc)
        
        # Predict labels using training set, record training accuracy
        train_pred = prediction(w_train, x_train)
        trainAcc = evaluate_accuracy(y_train, train_pred)
        trainAccuracies.append(trainAcc)

    return validAccuracies, trainAccuracies

def avg(list1):
    return round(sum(list1) / len(list1),4)

def gridSearch(x, y, K, model_name, preHyps, modelHyps, verbose):
    ''' 
    Constructs a grid with pre-processing hyperparameters (preHyps) and model-related hyperparameters (modelHyps). 
    Computes the K-fold cross-validation training and validation accuracy for each hyperparameter set.
    '''
    P = len(preHyps)
    Q = len(modelHyps)
    valAccs = np.zeros((P,Q,K))
    trainAccs = np.zeros((P,Q,K))

    for p in range(P):
        # Pre-process data for each set of pre-processing hyperparameters but do not remove outliers
        preHyp = preHyps[p]
        if verbose:
            print("Pre-processing hyperparameters:", str(preHyp))
        fullPreHyp = preHyp
        fullPreHyp['beta'] = None
        pre_x, pre_y = pre_process(x, y, **fullPreHyp)

        # Perform k-fold cross validation for each model hyperparameter
        for q in range(Q):
            modelHyp = modelHyps[q]
            valAcc, trainAcc = kfoldCV(pre_x, pre_y, K, model_name, modelHyp, beta=preHyp['beta'])
            valAccs[p,q,:] = valAcc
            trainAccs[p,q,:] = trainAcc
            if verbose: 
                print("\tModel hyperparameters:", str(modelHyp))
                print('\t\tTraining accuracies: ', trainAcc, '\taverage =', avg(trainAcc))
                print('\t\tValidation accuracies: ', valAcc, '\taverage =', avg(valAcc))
            
    return trainAccs, valAccs


def hyperparameterTuning(x, y, K, model_name, preHyps, modelHyps, verbose=False):
    '''
    Hyperparameter tuning function: Chooses hyperparameters with highest average
    validation accuracy and returns the test accuracy of the optimal model
    preHyps = list of dictionaries of pre-processing hyperparameters
    hyps = list of dictionaries of model hyperparameters
    '''
    # Compute K-fold cross-validation accuracies for each hyperparameter set
    trainAccs, valAccs = gridSearch(x, y, K, model_name, preHyps, modelHyps, verbose)

    # Find hyperparameter set with highest average validation accuracy
    avgValAccs = np.mean(valAccs, 2) 
    avgTrainAccs = np.mean(trainAccs, 2) 
    optP, optQ  = np.unravel_index(avgValAccs.argmax(), avgValAccs.shape)
    optPreHyp = preHyps[optP]
    optModelHyp = modelHyps[optQ]

    # Pre-process whole training set with optimal pre-processing hyperparameters
    print('Optimal pre-processing parameters: ', optPreHyp)
    x_train_opt, y_train_opt = pre_process(x=x, y=y, **optPreHyp)

    # Train model with optimal hyperparameters and whole training set
    print('Optimal model parameters: ', optModelHyp)
    wOpt, _ = model_name(y=y_train_opt, tx=x_train_opt, **optModelHyp)

    # Pre-process whole training set but keep outlier rows
    fullOptPreHyp = optPreHyp
    fullOptPreHyp['beta'] = None
    x_opt_full, _ = pre_process(x=x, y=y, **fullOptPreHyp)

    print('Average K-fold validation accuracy:', round(avgValAccs[optP, optQ], 4))
    print('Average K-fold training accuracy:', round(avgTrainAccs[optP, optQ], 4))

    # Obtain final prediction for whole training set
    optPred = prediction(wOpt, x_opt_full)
    optAccuracy = evaluate_accuracy(optPred, y)
    print('Full training accuracy for optimal hyperparameters: ', optAccuracy)
    return optPreHyp, optModelHyp, optAccuracy
               

    
