import numpy as np


def accuracy_score(y_true, y_pred):

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    predicted_positive = np.sum(y_pred == 1)

    return true_positive / predicted_positive


def recall_score(y_true, y_pred):

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    actual_positive = np.sum(y_true == 1)

    return true_positive / actual_positive