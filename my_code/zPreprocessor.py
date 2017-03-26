#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Preprocessor class by Gold team

"""

from sys import argv
from sklearn.base import BaseEstimator
import data_manager #The class provided by binome 1
# Note: if zDataManager is not ready, use the mother class DataManager
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = PCA(n_components=2)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        self.transformer.fit(X, y)
        # adding 'genCoef' feature
        X = np.hstack((X, np.subtract(X[:,51], X[:, 52]).reshape(105000, 1)))
        # we cannot have negative values for SelectKBest + standarization
        minmax = MinMaxScaler()
        X = minmax.fit_transform(X)
        # we select best k features
        selectBest = SelectKBest(chi2, k=30) #k is number of features.
        X = selectBest.fit_transform(X, y)
        return X


    def transform(self, X, y=None):
        return self.transformer.transform(X)

if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];

    basename = 'movierec'
    D = data_manager.DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D

    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D

