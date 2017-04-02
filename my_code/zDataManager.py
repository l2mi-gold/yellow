#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program that reads data and has a few display methods.
"""

# Add the sample code in the path
mypath = "../sample_code"
from sys import argv, path
from os.path import abspath
path.append(abspath(mypath))

import zPreprocessor

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# Data types
import pandas as pd

# Mother class
import data_manager

class DataManager(data_manager.DataManager):
    '''This class reads and displays data. 
       With class inheritance, we do not need to redefine the constructor,
       unless we want to add or change some data members.
       '''
	
    # Moyenne qu'une certaine catégorie de personne a donnée à un film ou à un genre de film (sexe, job, âge...)
    # Pour certaines catégorie il y a très peu de représentants (< 5), ce n'est donc pas très représentatif
    
    def CategoryAverageRating(self, userCategory, limit, movieId=0, movieGenre=''):
		if not {userCategory}.issubset(self.DF): return 0
		if not {movieGenre}.issubset(self.DF) and movieGenre != '' : return 0
		category = self.DF[userCategory] == 1
		if(movieGenre != ''):
			genre = self.DF[movieGenre] == 1
		film = self.DF['movie_id'] == movieId
		result = self.DF
		if(movieId == 0 and movieGenre == ''):
			result = self.DF[category]
		elif(movieGenre == ''):
			result = self.DF[category & film]
		elif(movieId == 0):
			result = self.DF[category & genre]
		else:
			result = self.DF[category & genre & film]
		total = 0
		for i in range(0, len(result)):
			total += result.iloc[i]['target']
		if len(result) == 0: return 0
		if len(result) <= limit: print("WARNING: Le nombre de représentant pour vos paramètres est inférieur ou égal à votre seuil")
		return total/len(result)
		
    def UserRatingMovieID(self, userId, movieId):
        film = self.DF['movie_id'] == movieId
        user = self.DF['user_id'] == userId
        
        if(len(self.DF[film & user]) > 0):
            return self.DF[film & user].iloc[0]['target']
        else:
            return 0 # l'utilisateur n'a pas noté ce film
    
    # Note moyenne d'un film
    
    def MovieAverageRating(self, movieId):
        film = self.DF['movie_id'] == movieId
        total = 0;
        for i in range(0, len(self.DF[film])):
            total += self.DF[film].iloc[i]['target']
        if len(self.DF[film]) == 0: return 0
        return total/len(self.DF[film])

    # Affiche le profil de l'utilisateur (note qu'a mis l'utilisateur aux films qu'il a vu par rapport à la note moyenne attribuée à ces films)
    
    def ShowUserProfile(self, userId):
        sort_data = self.DF.sort_values(by=['movie_id'], ascending=[True])
        user = sort_data['user_id'] == userId
        final_data = sort_data[user]
        plt.plot(final_data['movie_id'], final_data['target'], c='red')
        plt.plot(final_data['movie_id'], final_data['movie_average_rating'], c='blue')
        plt.legend(['User Rating', 'Movie Average Rating'])
        plt.show()
        return
    
    # Show Correlation Matrix before and after PCA
    
    def ShowCorrelationMatrix(self):
        Prepro = zPreprocessor.Preprocessor()
        # Don't change self
        D = self
        d = self.toDF('train')
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
        # Preprocess on the data and load it back into D (Apply PCA)
        D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
        D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
        D.data['X_test'] = Prepro.transform(D.data['X_test'])
        DF = pd.DataFrame(D.data['X_train'])
        Y = D.data['Y_train']
        DF = DF.assign(target=Y)
        corr1 = DF.corr()
        # Set up the matplotlib figure
        f1, ax1 = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap1 = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr1, cmap=cmap1, vmax=.3,
                    square=True, xticklabels=5, yticklabels=5,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)
        sns.plt.show()
        return
        
    def toDF(self, set_name):
        ''' Change a given data subset to a data Panda's frame.
            set_name is 'train', 'valid' or 'test'.'''
        DF = pd.DataFrame(self.data['X_'+set_name])
        # For training examples, we can add the target values as
        # a last column: this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        if set_name == 'train':
            Y = self.data['Y_train']
            DF = DF.assign(target=Y)          
        return DF
        
    def DataStats(self, set_name):
        ''' Display simple data statistics'''
        DF = self.toDF(set_name)
        print DF.describe()
    
    def ShowScatter(self, var1, var2, set_name):
        ''' Show scatter plots.'''
        DF = self.toDF(set_name)
        if set_name == 'train':
            sns.pairplot(DF.ix[:, [var1, var2, "target"]], hue="target")
        else:
            sns.pairplot(DF.ix[:, [var1, var2]])
    
if __name__=="__main__":
    # We can use this to run this file as a script and test the DataManager
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    
    basename = 'movierec'
    D = DataManager(basename, input_dir)
    
    print D