from sklearn.base import BaseEstimator
from zPreprocessor import Preprocessor

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pickle


class Regressor(BaseEstimator):
	def __init__(self):
		self.clf = VotingClassifier(estimators=[
                ('basic1', DecisionTreeRegressor()), 
                ('basic2', LinearRegression())], 
                voting='soft')  

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict(X)

	def predict_proba(self, X):
		return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes

	def get_classes(self):
		return self.clf.classes_

	def save(self, path="./"):
		pickle.dump(self, open(path + '_model.pickle', "w"))

	def load(self, path="./"):
		self = pickle.load(open(path + '_model.pickle'))
		return self
