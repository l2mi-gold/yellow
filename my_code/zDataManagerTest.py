#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program that test the Yellow challenge Data Manager class.
"""

from zDataManager import DataManager
input_dir = "../public_data"
output_dir = "../res"

basename = 'movierec'
D = DataManager(basename, input_dir)
print D
    
D.DataStats('train')
#D.ShowScatter(1, 2, 'train')
D.ShowUserProfile(5103)
D.ShowUserProfile(5113)
D.ShowUserProfile(4104)
D.ShowCorrelationMatrix()

print(D.CategoryAverageRating('age_18-24', 5, movieId=3174))

assert D.UserRatingMovieID(100000, 100000) == 0

# MovieAverageRating :

# Selection inexistante
assert D.MovieAverageRating(100000) == 0

# CategoryAverageRating :

# Category inexistante :
assert D.CategoryAverageRating('test', 5, movieId=3174) == 0
# Film inexistant :
assert D.CategoryAverageRating('job_other', 5, movieId=100000) == 0
# Genre inexistant :
assert D.CategoryAverageRating('age_18-24', 5, movieGenre='test') == 0
# Moyenne inférieur à 5
assert D.CategoryAverageRating('job_other', 5, movieId=3174) <= 5
# Moyenne supérieur à 0
assert D.CategoryAverageRating('job_other', 5, movieId=3174) >= 0