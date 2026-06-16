class BaseEstimator:

    def fit(self, X, y):
        raise NotImplementedError(
            "fit() method must be implemented"
        )

    def predict(self, X):
        raise NotImplementedError(
            "predict() method must be implemented"
        )