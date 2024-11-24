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
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the model

        Args:
            X: Input data (n_samples, n_features)
            y: Target data (n_samples, n_features)
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode the data to latent space

        Args:
            X: (n_samples, n_features)
        Returns:
            encodings: (n_samples, n_latent_dim)
        """
        pass

    @abstractmethod
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Decode the data from latent space

        Args:
            Z: Latent space representation (n_samples, n_latent_dim)
        Returns:
            X: (n_samples, n_features)
        """
        pass

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode and then decode the data"""
        return self.inverse_transform(self.transform(X))
