#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from zDataManager import DataManager
from zPreprocessor import Preprocessor
input_dir = "../public_data"
output_dir = "../res"

basename = 'movierec'
D = DataManager(basename, input_dir) # Load data
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
