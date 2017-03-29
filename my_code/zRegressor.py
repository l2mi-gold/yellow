from sklearn.base import BaseEstimator

# Pour DecisionTreeRegressor()
# from sklearn.tree import DecisionTreeRegressor

# Pour linear_model.SGDRegressor()
from sklearn import linear_model

# Pour BaggingRegressor()
# from sklearn.utils import check_random_state
# from sklearn.ensemble import BaggingRegressor

# For our preprocessor
from zPreprocessor import Preprocessor
from sys import argv
import pickle


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = linear_model.SGDRegressor()
        self.preprocessor = Preprocessor()

    def fit(self, X, y):
        # self.clf = DecisionTreeRegressor()
        # rng = check_random_state(0)
        # self.clf = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
        #                  n_estimators=20,
        #                  bootstrap=True,
        #                  oob_score=True,
        #                  random_state=rng)
        X = self.preprocessor.fit_transform(X.as_matrix(), y)
        self.clf.fit(X, y)

    def predict(self, X):
        X = self.preprocessor.transform(X.as_matrix())
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)  # The classes are in the order of the labels returned by get_classes

    def get_classes(self):
        return self.clf.classes_

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self


if __name__ == "__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];


    datadir = '../public_data/'  # Change this to the directory where you put the input data
    dataname = 'movierec'
    basename = datadir + dataname
    import data_io
    import eval

    codedir = 'my_code/'  # Change this to the directory where you put the code
    from sys import path;

    path.append(codedir)
    import seaborn as sns;

    sns.set()

    reload(data_io)
    data = data_io.read_as_df(basename)  # The data are loaded as a Pandas Data Frame

    import numpy as np

    X_train = data.drop('target', axis=1)  # This is the data matrix you already loaded (training data)
    y_train = data['target'].values  # These are the target values encoded as categorical variables
    print 'Dimensions X_train=', X_train.shape, 'y_train=', y_train.shape
    X_valid = data_io.read_as_df(basename, 'valid')

    X_test = data_io.read_as_df(basename, 'test')

    clf = Regressor()
    clf.fit(X_train, y_train)
    Y_valid = clf.predict(X_valid)
    Y_test = clf.predict(X_test)
    # clf.load(outname) # Uncomment to check reloading works
    data_io.write('score' + '_valid.predict', Y_valid)
    data_io.write('score' + '_test.predict', Y_test)
