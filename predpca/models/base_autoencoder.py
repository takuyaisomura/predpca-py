from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class BaseAutoEncoder(ABC):
    """Abstract base class for models that encode and decode data"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model"""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the model"""
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

    @abstractmethod
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent representations back to original space

        Args:
            Z: Latent representations (n_samples, n_latent_dim)
        Returns:
            X_reconstructed: Reconstructed data (n_samples, n_features)
        """
        pass

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode and then decode the data

        Args:
            X: Input data (n_samples, n_features)
        Returns:
            X_reconstructed: Reconstructed data (n_samples, n_features)
        """
        return self.decode(self.encode(X))
