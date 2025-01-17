from typing import Self

import numpy as np
from deeptime.decomposition import TICA as _TICA

from predpca.models.base_encoder import BaseEncoder


class TICA(BaseEncoder):
    def __init__(
        self,
        dim: int | None = None,
        lagtime: int = 1,
        scaling: str | None = "kinetic_map",
    ):
        super().__init__()
        self._tica = _TICA(
            dim=dim,
            lagtime=lagtime,
            scaling=scaling,
        )
        self._tica_model = None

    @property
    def name(self) -> str:
        return "TICA"

    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        self._tica_model = self._tica.fit((X, X_target)).fetch_model()
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self._tica_model.forward(X, propagate=True)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        # based on CovarianceKoopmanModel.propagate()
        dim = self._tica_model.output_dimension
        inv = np.linalg.pinv(self._tica_model._whitening_instantaneous.sqrt_inv_cov)[:dim]  # (dim, n_features)
        mean = self._tica_model._whitening_instantaneous.mean  # (n_features,)
        reconst = Z @ inv + mean

        return reconst
