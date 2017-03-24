#!/usr/bin/env python
# commentaire
# obtenir une représentation concise des utilisateurs qui ont aimés le plus un film donné
def UserProfileFilmID(data, FilmID):
    filmdb = data.groupby("movie_id")
    
    #meaneval = data.
    return;

codedir = 'sample_code/'                        # Change this to the directory where you put the code
from sys import path; path.append(codedir)
#%matplotlib inline
import seaborn as sns; sns.set()
datadir = 'public_data/'                        # Change this to the directory where you put the input data
dataname = 'movierec'
basename = datadir  + dataname
import data_io
#import eval
reload(data_io)
data = data_io.read_as_df(basename)             # The data are loaded as a Pandas Data Frame

#UserProfileFilmID(data, 321)
filmdb = data.groupby("movie_id")
filmdb.head()
