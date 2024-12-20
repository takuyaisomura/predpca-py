from typing import Self

import numpy as np
from sklearn.decomposition import PCA

from predpca.models.base_encoder import BaseEncoder
from predpca.models.predpca.model import PredPCA


class IdentityProcessor:
    def fit(self, X: np.ndarray) -> Self:
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


class PredPCAEncoder(BaseEncoder):
    def __init__(
        self,
        model: PredPCA,
        Ns: int,
        Nu: int,
        enable_preprocess: bool = True,
        enable_postprocess: bool = True,
    ):
        """Initialize PredPCA AutoEncoder

        Args:
            model: PredPCA model
            Ns: Number of state variables
            Nu: Number of latent variables
            enable_preprocess: Whether to enable preprocess PCA
            enable_postprocess: Whether to enable postprocess PCA
        """
        super().__init__()
        self.model = model
        self.preprocessor = PCA(n_components=Ns) if enable_preprocess else IdentityProcessor()
        self.postprocessor = PCA(n_components=Nu) if enable_postprocess else IdentityProcessor()

    @property
    def name(self) -> str:
        return "PredPCA"

    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        s = self.preprocessor.fit_transform(X)  # (n_samples, Ns)
        s_target = self.preprocessor.transform(X_target)  # (n_samples, Ns)
        s_pred = self.model.fit_transform(s.T, s_target.T).T  # (n_samples, Ns)
        self.postprocessor.fit(s_pred)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        s = self.preprocessor.transform(X)  # (n_samples, Ns)
        s_pred = self.model.transform(s.T).T  # (n_samples, Ns)
        return self.postprocessor.transform(s_pred)  # (n_samples, Nu)

    def decode(self, u: np.ndarray) -> np.ndarray:
        s_pred = self.postprocessor.inverse_transform(u)  # (n_samples, Ns)
        s = self.model.inverse_transform(s_pred.T).T  # (n_samples, Ns)
        return self.preprocessor.inverse_transform(s)  # (n_samples, n_features)
