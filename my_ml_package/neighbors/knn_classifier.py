import numpy as np

from my_ml_package.base.base_estimator import BaseEstimator

from my_ml_package.neighbors.distance import euclidean_distance
from my_ml_package.neighbors.voting import majority_vote


class KNNClassifier(BaseEstimator):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = []

        for x in X:

            distances = []

            for x_train in self.X_train:

                distance = euclidean_distance(x, x_train)

                distances.append(distance)

            k_indices = np.argsort(distances)[:self.k]

            k_nearest_labels = self.y_train[k_indices]

            prediction = majority_vote(k_nearest_labels)

            predictions.append(prediction)

        return np.array(predictions)