import numpy as np


# TODO: integrate with postprocessing.py
class WTAClassifier:
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using winner-takes-all

        Args:
            X: ICA transformed encodings (n_samples, n_classes)
        Returns:
            Winner-takes-all predictions (n_samples, n_classes)
        """
        return X == X.max(axis=1)[:, np.newaxis]  # (n_samples, n_classes)

    def compute_categorization_error(self, pred_onehot: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate categorization error

        Args:
            pred_onehot: Predicted one-hot labels (n_samples, n_classes)
            y_true: True labels (n_samples,)
        Returns:
            Categorization error
        """
        n_classes = y_true.max() + 1

        # Confusion matrix (true vs pred)
        G = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            G[i, :] = np.sum(pred_onehot[y_true == i], axis=0)

        return np.mean(1 - G.max(axis=0) / (G.sum(axis=0) + 0.001))
