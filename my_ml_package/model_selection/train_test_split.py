import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):

    n_samples = len(X)

    n_test = int(n_samples * test_size)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_indices = indices[:n_test]

    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test