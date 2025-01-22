from typing import Self

import numpy as np
from sklearn.decomposition import PCA

from predpca.models.base_encoder import BaseEncoder
from predpca.models.model import PredPCA


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
        Ns: int | None = None,
        Nu: int | None = None,
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

        if enable_preprocess:
            if Ns is None:
                raise ValueError("Ns must be provided if enable_preprocess is True")
            self.preprocessor = PCA(n_components=Ns)
        else:
            self.preprocessor = IdentityProcessor()

        if enable_postprocess:
            if Nu is None:
                raise ValueError("Nu must be provided if enable_postprocess is True")
            self.postprocessor = PCA(n_components=Nu)
        else:
            self.postprocessor = IdentityProcessor()

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
        s = self.preprocessor.fit_transform(X)
        s_target = self.preprocessor.transform(X_target)  # (n_samples, Ns)
        if X.ndim == 2:  # (n_samples, Ns)
            s_pred = self.model.fit_transform(s.T, s_target.T).T  # (n_samples, Ns)
        else:  # (Ns, n_seq, seq_len)
            s_pred = self.model.fit_transform(s, s_target.T).T  # (n_samples, Ns)
        self.postprocessor.fit(s_pred)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        s = self.preprocessor.transform(X)  # (n_samples, Ns)
        if X.ndim == 2:  # (n_samples, Ns)
            s_pred = self.model.transform(s.T).T  # (n_samples, Ns)
        else:  # (Ns, n_seq, seq_len)
            s_pred = self.model.transform(s).T  # (n_samples, Ns)
        return self.postprocessor.transform(s_pred)  # (n_samples, Nu)

    def decode(self, u: np.ndarray) -> np.ndarray:
        s_pred = self.postprocessor.inverse_transform(u)  # (n_samples, Ns)
        X_pred = self.preprocessor.inverse_transform(s_pred)  # (n_samples, n_features)
        return X_pred
