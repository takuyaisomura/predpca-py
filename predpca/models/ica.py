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

    def fit(
        self,
        X: np.ndarray,
        X_test: np.ndarray,
    ) -> Self:
        """Learn the ICA matrix and compute transformation parameters

        Args:
            X: array-like of shape (n_samples, n_features)
                encodings of training data
            X_test: array-like of shape (n_samples, n_features)
                encodings of test data for computing skewness signs

        Returns:
            self: object
                trained Transformer
        """
        self._fit(X, X_test)
        return self

    def fit_transform(self, X: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        X_ica = self._fit(X, X_test)
        return X_ica

    def _fit(self, X: np.ndarray, X_test: np.ndarray) -> None:
        X_t = X.T  # (n_features, n_samples)
        n_features, n_samples = X_t.shape

        # preprocess
        std = np.diag(np.std(X_t, axis=1, ddof=1))  # (n_features, n_features)
        self.pre_std_scaling_ = linalg.inv(std)  # (n_features, n_features)
        X_normalized = self.pre_std_scaling_ @ X_t  # (n_features, n_samples)

        # ICA learning
        rndn = np.random.randn(self.n_classes, n_features)
        Wica, _, _ = linalg.svd(rndn)

        sample_size = n_samples // 10
        for _ in tqdm(range(self.n_iterations), desc="ICA"):
            rnd = np.random.randint(n_samples, size=sample_size)
            X_ica = Wica @ X_normalized[:, rnd]  # (n_classes, sample_size)
            g = np.sqrt(2) * np.tanh(100 * X_ica)
            Wica += self.learning_rate * (Wica - (g @ X_ica.T / sample_size) @ Wica)

        self.Wica_ = Wica
        X_ica = self.Wica_ @ X_normalized  # (n_classes, n_samples)

        # Compute skewness signs using test data to adjust the sign of components
        X_test_ica = self.Wica_ @ self.pre_std_scaling_ @ X_test.T  # (n_classes, n_samples)
        self.skewness_signs_ = np.sign(stats.skew(X_test_ica, axis=1, keepdims=True))  # (n_classes, 1)

        # postprocess
        X_ica *= self.skewness_signs_
        transformed_std = np.diag(np.std(X_ica, axis=1, ddof=1))
        self.post_std_scaling_ = np.sqrt(0.1) * linalg.inv(transformed_std)
        X_ica = self.post_std_scaling_ @ X_ica  # (n_classes, n_samples)

        return X_ica.T  # (n_samples, n_classes)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the learned ICA matrix with pre/post processing

        Args:
            X: array-like of shape (n_samples, n_features)
                encodings of data to transform

        Returns:
            X_transformed: array-like of shape (n_samples, n_features)
                transformed encodings
        """
        X_normalized = self.pre_std_scaling_ @ X.T  # (n_features, n_samples)
        X_ica = self.Wica_ @ X_normalized  # (n_classes, n_samples)
        X_ica *= self.skewness_signs_
        X_ica = self.post_std_scaling_ @ X_ica  # (n_classes, n_samples)
        return X_ica.T  # (n_samples, n_classes)
