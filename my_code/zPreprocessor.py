#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Preprocessor class by Gold team

"""

from sys import argv
from sklearn.base import BaseEstimator
import zDataManager #The class provided by binome 1
# Note: if zDataManager is not ready, use the mother class DataManager
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocessor(BaseEstimator):
 def __init__(self):
        self.transformer = PCA(n_components=20)
        self.selector = SelectKBest(chi2, k=30) #k is number of features.
        self.minmax = MinMaxScaler()

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        # adding 'genCoef' feature
        X = np.hstack((X, np.divide(X[:,51], X[:, 52]).reshape(X.shape[0], 1)))
        #  standarization
        X = self.minmax.fit_transform(X)
        # we select best k features
        X = self.selector.fit_transform(X, y)
        # on applique PCA
        self.transformer.fit(X, y)
        return self.transformer.transform(X)


    def transform(self, X, y=None):
        X = np.hstack((X, np.divide(X[:,51], X[:, 52]).reshape(X.shape[0], 1)))
        X = self.minmax.transform(X)
        X = self.selector.transform(X)
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
    D = zDataManager.DataManager(basename, input_dir) # Load data
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

