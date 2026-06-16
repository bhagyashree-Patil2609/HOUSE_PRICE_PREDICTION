import numpy as np

class PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Identify columns with finite values and non-zero variance, keep mask for transform
        finite_mask = np.all(np.isfinite(X_centered), axis=0)
        # Compute variance only on finite columns
        var = np.var(X_centered[:, finite_mask], axis=0) if np.any(finite_mask) else np.array([])
        nonzero_submask = var > 1e-12

        # Build full feature mask aligned with original columns
        feature_mask = np.zeros(X.shape[1], dtype=bool)
        if np.any(finite_mask):
            feature_mask[np.where(finite_mask)[0]] = nonzero_submask

        self._feature_mask = feature_mask

        # Select the valid columns for covariance computation
        X_centered_sel = X_centered[:, self._feature_mask]
        covariance_matrix = np.cov(X_centered_sel.T)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        except np.linalg.LinAlgError:
            # Fall back to SVD-based computation for numerical stability
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            eigenvectors = Vt.T
            eigenvalues = (S ** 2) / (X_centered.shape[0] - 1)

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros_like(eigenvalues)
        else:
            self.explained_variance_ratio_ = eigenvalues / total_variance

        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumulative_ratio = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = int(np.searchsorted(cumulative_ratio, self.n_components) + 1)
        else:
            self.n_components_ = self.n_components if self.n_components is not None else X.shape[1]

        self.components = eigenvectors[:, :self.n_components_].T

    def transform(self, X):
        X_centered = X - self.mean_
        # Apply same feature selection used during fit
        if hasattr(self, "_feature_mask"):
            X_centered = X_centered[:, self._feature_mask]
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
