import numpy as np


# TODO: integrate with postprocessing.py
class WTAClassifier:
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using winner-takes-all

        Args:
            X: ICA transformed encodings (n_samples, n_classes)
        Returns:
            predictions: Winner-takes-all predictions (n_samples, n_classes)
        """
        predictions = X == X.max(axis=1)[:, np.newaxis]
        return predictions

    def compute_categorization_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate categorization error

        Args:
            X: ICA transformed encodings (n_samples, n_classes)
            y: True labels (n_samples,)
        Returns:
            Categorization error
        """
        _, n_classes = X.shape
        predictions = self.predict(X)  # (n_samples, n_classes)

        # Confusion matrix (true vs pred)
        G = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            G[i, :] = np.sum(predictions[y == i], axis=0)

        return np.mean(1 - G.max(axis=0) / (G.sum(axis=0) + 0.001))
