# ml-project-1-golinearorgohome
## EPFL: CS-433 Machine Learning - Project 1
### Detecting the Higgs Boson: A ML Challenge

**Team**: GoLinearOrGoHome (Antoine Bonnet, Melanie Brtan, Camille Cathala)

We obtained a test accuracy of 78.6% with the Ridge regression model. 

To run this project: 
1. Download the train and test dataset files from [Kaggle](https://www.kaggle.com/competitions/higgs-boson/data).
2. Place the files in the **data** directory. 
3. Run the script **run.py**. 

The following files can be found in this repository: 

**scripts/implementations.py** contains methods for training models such as least squares regression, ridge regression, gradient descent, stochastic gradient descent,  logistic regression and regularized logistic regression. 

**scripts/load_data.py** contains methods for loading the data and pre-processing it, including missing value replacement, removing outliers, polynomial expansion and standardization/normalization.  

**scripts/train.py** contains methods for model training including K-fold cross-validation and hyperparameter tuning. 

**scripts/experiment.ipynb** contains our experiments with model training and hyperparameter tuning. 

**scripts/helpers.py** contains some helper functions for loading data from a csv file.  

**scripts/run.py** is used to produce the final test labels, once we have selected our optimal model. 
