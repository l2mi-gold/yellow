import numpy as np
from zDataManager import DataManager
from zRegressor import Regressor
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score

input_dir = "../public_data"
output_dir = "../res"

basename = 'movierec'
D = DataManager(basename, input_dir)
print D

myregressor = Regressor()
 
Ytrue_tr = D.data['Y_train']
myregressor.fit(D.data['X_train'], Ytrue_tr)

Ypred_tr = myregressor.predict(D.data['X_train'])
Ypred_va = myregressor.predict(D.data['X_valid'])
Ypred_te = myregressor.predict(D.data['X_test'])  

acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)

acc_cv = cross_val_score(myregressor, D.data['X_train'], Ytrue_tr, cv=5, scoring='accuracy')

print "Training Accuracy = %5.2f +-%5.2f" % (acc_tr.mean(), acc_tr.std())
print "Cross-validation Accuracy = %5.2f +-%5.2f" % (acc_cv.mean(), acc_cv.std())
