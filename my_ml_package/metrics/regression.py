import numpy as np


def mse(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):

    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):

    ss_total = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    ss_residual = np.sum(
        (y_true - y_pred) ** 2
    )

    return 1 - (ss_residual / ss_total)

def adjusted_r2_score(y_true, y_pred, n_features):

    n = len(y_true)

    r2 = r2_score(y_true, y_pred)

    adjusted_r2 = 1 - (
        ((1 - r2) * (n - 1))
        / (n - n_features - 1)
    )

    return adjusted_r2