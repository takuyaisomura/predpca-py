from typing import Self

import numpy as np
from scipy import linalg, stats
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


# TODO: integrate with postprocessing.py
class ICA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_classes: int,
        n_iterations: int = 2000,
        learning_rate: float = 0.01,
    ):
        self.n_classes = n_classes
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.Wica_: np.ndarray | None = None  # placeholder for the learned ICA matrix
        self.pre_std_scaling_: np.ndarray | None = None
        self.post_std_scaling_: np.ndarray | None = None
        self.skewness_signs_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Learn the ICA matrix with pre/post processing

        Args:
            X: array-like of shape (n_samples, n_features)
                encodings of training data
            y: None
                ignored (for API compatibility)

        Returns:
            self: object
                trained Transformer
        """
        self._fit(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        transformed = self._fit(X)
        return transformed

    def _fit(self, X: np.ndarray) -> None:
        X_t = X.T  # (n_features, n_samples)
        n_features, n_samples = X_t.shape

        # preprocess
        std = np.diag(np.std(X_t, axis=1, ddof=1))  # (n_features, n_features)
        self.pre_std_scaling_ = linalg.inv(std)  # (n_features, n_features)
        X_normalized = self.pre_std_scaling_ @ X_t  # (n_features, n_samples)

        # ICA learning
        rnd = np.random.randn(self.n_classes, n_features)
        Wica, _, _ = linalg.svd(rnd)

        sample_size = n_samples // 10
        for _ in tqdm(range(self.n_iterations), desc="ICA"):
            rnd = np.random.randint(n_samples, size=sample_size)
            ui_train = Wica @ X_normalized[:, rnd]
            g = np.sqrt(2) * np.tanh(100 * ui_train)
            Wica += self.learning_rate * (Wica - (g @ ui_train.T / sample_size) @ Wica)

        self.Wica_ = Wica
        transformed = self.Wica_ @ X_normalized  # (n_classes, n_samples)

        # postprocess
        self.skewness_signs_ = np.sign(stats.skew(transformed, axis=0))
        transformed *= self.skewness_signs_

        transformed_std = np.diag(np.std(transformed, axis=1, ddof=1))
        self.post_std_scaling_ = np.sqrt(0.1) * linalg.inv(transformed_std)
        transformed = self.post_std_scaling_ @ transformed  # (n_classes, n_samples)

        return transformed.T  # (n_samples, n_classes)

    def transform(self, X):
        """Transform the data using the learned ICA matrix with pre/post processing

        Args:
            X: array-like of shape (n_samples, n_features)
                encodings of data to transform

        Returns:
            X_transformed: array-like of shape (n_samples, n_features)
                transformed encodings
        """
        X_t = X.T  # (n_features, n_samples)

        # preprocess
        X_normalized = self.pre_std_scaling_ @ X_t  # (n_features, n_samples)

        # ICA transform
        transformed = self.Wica_ @ X_normalized  # (n_classes, n_samples)

        # postprocess
        transformed *= self.skewness_signs_
        transformed = self.post_std_scaling_ @ transformed  # (n_classes, n_samples)

        return transformed.T  # (n_samples, n_classes)
