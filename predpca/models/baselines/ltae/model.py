from typing import Self

import numpy as np
from scipy import linalg


class LTAEModel:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.E: np.ndarray | None = None  # encoding matrix (n_features, n_components)
        self.D: np.ndarray | None = None  # decoding matrix (n_components, n_features)

    def fit(self, X: np.ndarray, X_target: np.ndarray) -> Self:
        """Fit the model to the data.

        Args:
            X: Input data of shape (n_samples, n_features)
            X_target: Target data of shape (n_samples, n_features)
        """
        n_samples, _ = X.shape

        # covariance matrix
        Sigmas = X.T @ X / n_samples  # (n_features, n_features)
        # regularization
        eps = 1e-6
        Sigmas += eps * np.eye(Sigmas.shape[0])
        # sqrt of inverse and sqrt of covariance matrix
        Sigmas_inv_sqrt = linalg.sqrtm(linalg.inv(Sigmas)).real  # (n_features, n_features)
        Sigmas_sqrt = linalg.sqrtm(Sigmas).real  # (n_features, n_features)

        # compute encoding and decoding matrices
        x = X @ Sigmas_inv_sqrt  # (n_samples, n_features)
        y = X_target @ Sigmas_inv_sqrt  # (n_samples, n_features)
        K = x.T @ y / n_samples  # (n_features, n_features)
        U, s, Vh = linalg.svd(K, full_matrices=False)
        # U: (n_features, N)
        # s: (N,)
        # Vh: (N, n_features)
        S = np.diag(s[: self.n_components])  # (n_components, n_components)

        self.E = Sigmas_inv_sqrt @ U[:, : self.n_components] @ S  # (n_features, n_components)
        self.D = Vh[: self.n_components] @ Sigmas_sqrt  # (n_components, n_features)

        return self

    def encode(self, s: np.ndarray) -> np.ndarray:
        """Transform data to latent space.

        Args:
            s: Input data of shape (n_samples, n_features)
        Returns:
            u: Latent space data of shape (n_samples, n_components)
        """
        if self.E is None:
            raise ValueError("Encoding matrix is not initialized")
        return s @ self.E

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Transform from latent space back to original space.

        Args:
            u: Latent space data of shape (n_samples, n_components)
        Returns:
            s: Original space data of shape (n_samples, n_features)
        """
        if self.D is None:
            raise ValueError("Decoding matrix is not initialized")
        return u @ self.D
