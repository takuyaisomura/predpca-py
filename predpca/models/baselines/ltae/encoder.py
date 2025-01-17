from typing import Self

import numpy as np

from predpca.models.base_encoder import BaseEncoder
from predpca.models.baselines.ltae.model import LTAEModel


class LTAE(BaseEncoder):
    def __init__(
        self,
        model: LTAEModel,
    ):
        super().__init__()
        self.model = model

    @property
    def name(self) -> str:
        return "LTAE"

    def fit(
        self,
        X: np.ndarray,
        X_target: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        X_target_val: np.ndarray | None = None,
    ) -> Self:
        self.model.fit(X, X_target)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.model.encode(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return self.model.decode(Z)
