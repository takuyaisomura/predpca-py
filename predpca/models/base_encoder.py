from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for models that encode and decode data"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model"""
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        """Train the model

        Args:
            X: Input data (n_samples, n_features)
            X_target: Target data (n_samples, n_features)
            X_val: Validation data (n_samples, n_features)
            X_target_val: Target data for validation (n_samples, n_features)
        """
        pass

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode the data to latent space

        Args:
            X: (n_samples, n_features)
        Returns:
            encodings: (n_samples, n_latent_dim)
        """
        pass

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode the data from latent space

        Args:
            Z: (n_samples, n_latent_dim)
        Returns:
            X_decoded: (n_samples, n_features)
        """
        # We don't make this abstract because not all models have a decode method
        raise NotImplementedError

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct the data from the original space

        Args:
            X: (n_samples, n_features)
        Returns:
            X_reconstructed: (n_samples, n_features)
        """
        return self.decode(self.encode(X))
