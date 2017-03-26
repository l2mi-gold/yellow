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

import matplotlib.pyplot as plt

# Graphic routines
import seaborn as sns; sns.set()

# Data types
import pandas as pd

# Mother class
import data_manager

class DataManager(data_manager.DataManager):
    '''This class reads and displays data. 
       With class inheritance, we do not need to redefine the constructor,
       unless we want to add or change some data members.
       '''

    def UserRatingMovieID(self, userId, movieId):
        film = self.DF['movie_id'] == movieId
        user = self.DF['user_id'] == userId
        
        if(len(self.DF[film & user]) > 0):
            return self.DF[film & user].iloc[0]['target']
        else:
            return 0 # l'utilisateur n'a pas noté ce film (trouver autre chose que 0)
    
    # Note moyenne d'un film
    
    def MovieAverageRating(self, movieId):
        film = self.DF['movie_id'] == movieId
        total = 0;
        for i in range(0, len(self.DF[film])):
            total += self.DF[film].iloc[i]['target']
        if len(self.DF[film]) == 0: return 0
        return total/len(self.DF[film])
    
    # Moyenne qu'une certaine catégorie de personne a donnée à un film (sexe, job, âge...)
    # Pour certaines catégorie il y a très peu de représentants (< 5), ce n'est donc pas très représentatif
    
    def CategoryAverageRatingMovieID(self, userCategory, movieId):
        if not {userCategory}.issubset(self.DF): return 0
        category = self.DF[userCategory] == 1
        film = self.DF['movie_id'] == movieId
        total = 0;
        for i in range(0, len(self.DF[category & film])):
            total += self.DF[category & film].iloc[i]['target']
        if len(self.DF[category & film]) == 0: return 0
        return total/len(self.DF[category & film])

    # Moyenne qu'une certaine catégorie de personne a donnée à une genre de film
    # Pour certaines catégorie il y a très peu de représentants (< 5), ce n'est donc pas très représentatif
    
    def CategoryAverageRatingMovieGenre(self, userCategory, movieGenre):
        if not {userCategory}.issubset(self.DF): return 0
        if not {movieGenre}.issubset(self.DF): return 0
        category = self.DF[userCategory] == 1
        genre = self.DF[movieGenre] == 1
        total = 0;
        for i in range(0, len(self.DF[category & genre])):
            total += self.DF[category & genre].iloc[i]['target']
        if len(self.DF[category & genre]) == 0: return 0
        return total/len(self.DF[category & genre])

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
    
#    def __init__(self, basename="", input_dir=""):
#        ''' New contructor.'''
#        DataManager.__init__(self, basename, input_dir)
        # So something here
    
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
    
    D.DataStats('train')
    D.ShowScatter(1, 2, 'train')
    D.ShowUserProfile(5103)