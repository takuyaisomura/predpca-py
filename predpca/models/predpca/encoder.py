from typing import Self

import numpy as np
from sklearn.decomposition import PCA

from predpca.models.base_encoder import BaseEncoder
from predpca.models.predpca.model import PredPCA


class PredPCAEncoder(BaseEncoder):
    def __init__(
        self,
        model: PredPCA,
        Ns: int,
        Nu: int,
    ):
        """Initialize PredPCA AutoEncoder

        Args:
            model: PredPCA model
            Ns: Number of state variables
            Nu: Number of latent variables
        """
        super().__init__()
        self.model = model
        self.Ns = Ns
        self.Nu = Nu
        self.pca_preprocess: PCA | None = None
        self.pca_postprocess: PCA | None = None

    @property
    def name(self) -> str:
        return "PredPCA"

    def fit(
        self,
        X: np.ndarray,
        X_val: np.ndarray | None = None,
    ) -> Self:
        # preprocess
        self.pca_preprocess = PCA(n_components=self.Ns)
        s = self.pca_preprocess.fit_transform(X).T  # (Ns, n_samples)

        s_pred = self.model.fit_transform(s, s)  # (Ns, n_samples)

        # postprocess
        self.pca_postprocess = PCA(n_components=self.Nu)
        self.pca_postprocess.fit(s_pred.T)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to latent space using PCA on predicted covariance

        Args:
            X: Input data (n_samples, input_dim)
        Returns:
            u: Latent representations (n_samples, latent_dim)
        """
        s = self.pca_preprocess.transform(X).T  # (Ns, n_samples)
        s_pred = self.model.transform(s)  # (Ns, n_samples)
        u = self.pca_postprocess.transform(s_pred.T)  # (n_samples, latent_dim)
        return u

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Decode latent representations back to input space

        Args:
            u: Latent representations (n_samples, latent_dim)
        Returns:
            X: Reconstructed data (n_samples, input_dim)
        """
        s_pred = self.pca_postprocess.inverse_transform(u).T  # (Ns, n_samples)
        s = self.model.inverse_transform(s_pred)  # (Ns, n_samples)
        X = self.pca_preprocess.inverse_transform(s.T)  # (n_samples, input_dim)
        return X
