from typing import Self

import numpy as np
from sklearn.decomposition import PCA

from predpca.models.base_encoder import BaseEncoder
from predpca.models.predpca.model import PredPCA


class PredPCAEncoder(BaseEncoder):
    def __init__(
        self,
        Ns: int,
        Nu: int,
        kp_list: list[int] | range,
        prior_s_: float,
        gain: np.ndarray | None = None,
    ):
        """Initialize PredPCA AutoEncoder

        Args:
            kp_list: List of time lags to use for prediction
            prior_s_: Prior variance for regularization
            gain: Optional gain matrix (Nu, Ns)
        """
        super().__init__()
        self.model = PredPCA(kp_list=kp_list, prior_s_=prior_s_, gain=gain)
        self.Ns = Ns
        self.Nu = Nu
        self.pca_preprocess: PCA | None = None
        self.pca_postprocess: PCA | None = None

    @property
    def name(self) -> str:
        return "PredPCA"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the model

        Args:
            X: Input data (n_samples, n_features)
            y: Not used
        """
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
