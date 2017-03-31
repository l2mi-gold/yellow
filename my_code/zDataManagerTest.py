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

assert D.UserRatingMovieID(100000, 100000) == 0

# MovieAverageRating :

# Selection inexistante
assert D.MovieAverageRating(100000) == 0

# CategoryAverageRatingMovieID :

# Category inexistante :
assert D.CategoryAverageRatingMovieID('test', 3174) == 0
# Film inexistant :
assert D.CategoryAverageRatingMovieID('job_other', 100000) == 0
# Moyenne inférieur à 5
assert D.CategoryAverageRatingMovieID('job_other', 3174) <= 5
# Moyenne supérieur à 0
assert D.CategoryAverageRatingMovieID('job_other', 3174) >= 0

# CategoryAverageRatingMovieGenre :

# Category inexistante :
assert D.CategoryAverageRatingMovieGenre('test', 'movie_genre_Sci-Fi') == 0
# Genre inexistant :
assert D.CategoryAverageRatingMovieGenre('age_18-24', 'test') == 0
# Moyenne inférieur à 5
assert D.CategoryAverageRatingMovieGenre('age_18-24', 'movie_genre_Sci-Fi') <= 5
# Moyenne supérieur à 0
assert D.CategoryAverageRatingMovieGenre('age_18-24', 'movie_genre_Sci-Fi') >= 0