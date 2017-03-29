from sklearn.base import BaseEstimator

# Pour DecisionTreeRegressor()
#from sklearn.tree import DecisionTreeRegressor

# Pour linear_model.SGDRegressor()
#from sklearn import linear_model

# Pour BaggingRegressor()
#from sklearn.utils import check_random_state
#from sklearn.ensemble import BaggingRegressor

# Pour AdaBoostRegressor()
#import numpy as np
#from sklearn.ensemble import AdaBoostRegressor

# Pour ExtraTreesRegressor()
#from sklearn.ensemble import ExtraTreesRegressor

# Pour RandomForestRegressor()
from sklearn.ensemble.forest import RandomForestRegressor

import pickle


class Regressor(BaseEstimator):
	def __init__(self):
		pass

	def fit(self, X, y):
		#self.clf = DecisionTreeRegressor()

		#self.clf = linear_model.SGDRegressor()

		#rng = check_random_state(0)
		#self.clf = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
         #                  n_estimators=5,
         #                  bootstrap=True,
         #                  oob_score=True,
         #                  random_state=rng)

		#rng = np.random.RandomState(1)
		#self.clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
		#			n_estimators=100, random_state=rng)

		#self.clf = ExtraTreesRegressor(20)

		self.clf = RandomForestRegressor(20)

		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		# The classes are in the order of the labels returned by get_classes
		return self.clf.predict_proba(X)

	def get_classes(self):
		return self.clf.classes_

	def save(self, path="./"):
		pickle.dump(self, open(path + '_model.pickle', "w"))

	def load(self, path="./"):
		self = pickle.load(open(path + '_model.pickle'))
		return self
