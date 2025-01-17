import numpy as np
from scipy import linalg


class Decoder:
    def __init__(
        self,
        prior_u: float,
        input_mean: np.ndarray,  # (n_features,)
    ):
        self.prior_u = prior_u
        self.input_mean = input_mean
        self.A: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,  # (n_samples, n_features)
        ica_encodings: np.ndarray,  # (n_samples, n_classes)
    ):
        _, n_classes = ica_encodings.shape
        self.A = (
            linalg.inv(ica_encodings.T @ ica_encodings + np.eye(n_classes) * self.prior_u) @ ica_encodings.T @ X
        )  # (n_classes, n_features)
        return self

    def transform(
        self,
        pred_onehot: np.ndarray,  # (n_samples, n_classes)
    ) -> np.ndarray:
        return pred_onehot @ self.A + self.input_mean  # (n_samples, n_features)
