import numpy as np

from my_ml_package.base.base_estimator import BaseEstimator
from my_ml_package.neighbors.distance import euclidean_distance


class KNNRegressor(BaseEstimator):

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before calling predict().")

        predictions = []

        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k = min(self.k, len(self.X_train))
            k_indices = np.argsort(distances)[:k]
            k_nearest_targets = self.y_train[k_indices]
            prediction = np.mean(k_nearest_targets)
            predictions.append(prediction)

        return np.array(predictions)
