from typing import Self

import numpy as np
import torch
from torch import Tensor


def matrix_sqrt(A: Tensor) -> Tensor:
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    S_sqrt = torch.sqrt(S)
    return U @ torch.diag(S_sqrt) @ V


def matrix_sqrt_inv(A: Tensor) -> Tensor:
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    S_inv_sqrt = 1 / torch.sqrt(S)
    return U @ torch.diag(S_inv_sqrt) @ V


class LTAEModel:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.E: Tensor | None = None  # encoding matrix (n_features, n_components)
        self.D: Tensor | None = None  # decoding matrix (n_components, n_features)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, X_target: np.ndarray) -> Self:
        """Fit the model to the data.

        Args:
            X: Input data of shape (n_samples, n_features)
            X_target: Target data of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Convert to torch tensors
        X_tensor = torch.from_numpy(X).to(self.device)
        X_target_tensor = torch.from_numpy(X_target).to(self.device)

        # covariance matrix
        Sigmas = X_tensor.T @ X_tensor / n_samples  # (n_features, n_features)
        # matrix square root with regularization
        eps = 1e-6
        Sigmas += eps * torch.eye(n_features, device=self.device)
        Sigmas_sqrt = matrix_sqrt(Sigmas)
        Sigmas_inv_sqrt = matrix_sqrt_inv(Sigmas)

        # compute encoding and decoding matrices
        x = X_tensor @ Sigmas_inv_sqrt  # (n_samples, n_features)
        y = X_target_tensor @ Sigmas_inv_sqrt  # (n_samples, n_features)
        K = x.T @ y / n_samples  # (n_features, n_features)
        U, s, Vh = torch.linalg.svd(K, full_matrices=False)
        # U: (n_features, N)
        # s: (N,)
        # Vh: (N, n_features)
        S = torch.diag(s[: self.n_components])  # (n_components, n_components)
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
        s_tensor = torch.from_numpy(s).to(self.device)
        return (s_tensor @ self.E).cpu().numpy()

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Transform from latent space back to original space.

        Args:
            u: Latent space data of shape (n_samples, n_components)
        Returns:
            s: Original space data of shape (n_samples, n_features)
        """
        if self.D is None:
            raise ValueError("Decoding matrix is not initialized")
        u_tensor = torch.from_numpy(u).to(self.device)
        return (u_tensor @ self.D).cpu().numpy()
