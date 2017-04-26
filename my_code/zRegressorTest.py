# Add the sample code in the path
mypath = "../sample_code"
from sys import path
from os.path import abspath
path.append(abspath(mypath))

#import numpy as np
from zDataManager import DataManager
from zRegressor import Regressor
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error

input_dir = "../public_data"
output_dir = "../res"

basename = 'movierec'
D = DataManager(basename, input_dir)
print D

myregressor = Regressor()
 
Ytrue_tr = D.data['Y_train']
myregressor.fit(D.data['X_train'], Ytrue_tr, 6)

Ypred_tr = myregressor.predict(D.data['X_train'])
Ypred_va = myregressor.predict(D.data['X_valid'])
Ypred_te = myregressor.predict(D.data['X_test'])  

acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)
acc_cv = cross_val_score(myregressor, D.data['X_train'], Ytrue_tr, cv=5, scoring='accuracy')

print "Training Accuracy = %5.2f +-%5.2f" % (acc_tr.mean(), acc_tr.std())
print "Cross-validation Accuracy = %5.2f +-%5.2f" % (acc_cv.mean(), acc_cv.std())

##### Code importe du notebook

datadir = '../public_data/'
dataname = 'movierec'
basename = datadir  + dataname
import data_io
#import eval
reload(data_io)
data = data_io.read_as_df(basename)
# Data matrix you already loaded (training data)
X_train = data.drop('target', axis=1)
# Target values encoded as categorical variables
y_train = data['target'].values
print 'Dimensions X_train=', X_train.shape, 'y_train=', y_train.shape
X_valid = data_io.read_as_df(basename, 'valid')
X_test = data_io.read_as_df(basename, 'test')
# This is just an example of 2-fold cross-validation
skf = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.5, random_state=0)
i=0
for idx_t, idx_v in skf:
    i=i+1
    Xtr = X_train.iloc[idx_t]
    Ytr = y_train[idx_t]
    Xva = X_train.iloc[idx_v]
    Yva = y_train[idx_v]
    clf = Regressor()
    clf.fit(Xtr, Ytr, 1, 1)
    Y_predict = clf.predict(Xva)
    print 'Fold', i, 'mae = ', mean_absolute_error(Yva, Y_predict), ', mad = ', eval.mae(Y_predict, Yva)
    # print 'Fold', i, 'validation accuracy (MAE) = ', eval.mae(Y_predict, Yva)
