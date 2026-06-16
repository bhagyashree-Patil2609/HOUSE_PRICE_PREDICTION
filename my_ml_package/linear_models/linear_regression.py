import numpy as np

from my_ml_package.base.base_estimator import BaseEstimator
from my_ml_package.utils.validation import check_array


class LinearRegression(BaseEstimator):

    def __init__(self):

        self.theta = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):

        check_array(X)
        check_array(y)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        A = X_b.T.dot(X_b)
        b = X_b.T.dot(y)

        try:
            self.theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            ridge = 1e-8
            self.theta = np.linalg.solve(A + ridge * np.eye(A.shape[0]), b)

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def predict(self, X):

        check_array(X)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b.dot(self.theta)