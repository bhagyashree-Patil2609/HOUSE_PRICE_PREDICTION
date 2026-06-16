import numpy as np

def check_array(X):

    if not isinstance(X, np.ndarray):

        raise ValueError(
            "Input must be a NumPy array"
        )

    return True