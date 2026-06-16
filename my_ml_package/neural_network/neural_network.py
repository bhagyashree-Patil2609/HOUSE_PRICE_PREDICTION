import numpy as np

from my_ml_package.base.base_estimator import BaseEstimator

from my_ml_package.neural_network.activations import (
    sigmoid,
    sigmoid_derivative
)

class NeuralNetwork(BaseEstimator):

    def __init__(self, input_size, hidden_size, output_size, random_state=None):

        rng = np.random.default_rng(random_state)
        self.weights1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.weights2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):

        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)

        output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)

        return output

    def fit(self, X, y, epochs=1000, learning_rate=0.1):

        for _ in range(epochs):

            output = self.forward(X)

            error = y - output

            d_output = error * sigmoid_derivative(output)

            error_hidden = d_output.dot(self.weights2.T)

            d_hidden = error_hidden * sigmoid_derivative(self.hidden)

            self.weights2 += self.hidden.T.dot(d_output) * learning_rate
            self.bias2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.weights1 += X.T.dot(d_hidden) * learning_rate
            self.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def predict(self, X):

        output = self.forward(X)

        return np.round(output)