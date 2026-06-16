import numpy as np

from my_ml_package.base.base_estimator import BaseEstimator
from my_ml_package.utils.validation import check_array


class StandardScaler(BaseEstimator):

    def __init__(self):

        self.mean = None
        self.std = None

    def fit(self, X):

        check_array(X)

        self.mean = np.mean(X, axis=0)

        self.std = np.std(X, axis=0)

        # IMPORTANT FIX
        self.std[self.std == 0] = 1

    def transform(self, X):

        check_array(X)

        return (X - self.mean) / self.std

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)