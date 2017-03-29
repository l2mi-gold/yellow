#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from zDataManager import DataManager
from zPreprocessor import Preprocessor
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
input_dir = "../public_data"
output_dir = "../res"

basename = 'movierec'
D = DataManager(basename, input_dir) # Load data
print("*** Original data ***")
print(D.data['Y_train'])


d = D.toDF('train')
print type(d)

corr = d.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))


sns.set(style="white")

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

# sns.plt.show()


Prepro = Preprocessor()
 
# Preprocess on the data and load it back into D
D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
# Here show something that proves that the preprocessing worked fine
print("*** Transformed data ***")
DF = pd.DataFrame(D.data['X_train'])
Y = D.data['Y_train']
DF = DF.assign(target=Y)
corr1 = DF.corr()
print(corr1)

# Set up the matplotlib figure
f1, ax1 = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap
cmap1 = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr1, cmap=cmap1, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)

sns.plt.show()
print type(D.data['X_train'])
