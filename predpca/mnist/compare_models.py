from pathlib import Path

import numpy as np

from predpca.mnist.create_digit_sequence import create_digit_sequence
from predpca.models.base_autoencoder import BaseAutoEncoder
from predpca.models.baselines.vae.vae import VAE
from predpca.models.wta_classifier import ICA, WTAClassifier

mnist_dir = Path(__file__).parent
data_dir = mnist_dir / "data"
out_dir = mnist_dir / "output" / "model_comparison"


def main():
    results = compare_models(data_dir)

    # Display the results
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")


def compare_models(
    data_dir: Path,
    sequence_type: int = 1,
    train_size: int = 100000,
    test_size: int = 100000,
):
    # Prepare data
    input_train, input_test, _, _, label_test, _ = create_digit_sequence(
        data_dir,
        sequence_type,
        train_size,
        test_size,
        test_size,
        train_randomness=True,
        test_randomness=False,
        train_signflip=True,
        test_signflip=False,
    )
    input_train = input_train.T  # (n_features, n_samples)
    input_test = input_test.T  # (n_features, n_samples)
    label_test = label_test.ravel()  # (n_samples,)

    # Prepare models
    models = [
        VAE(epochs=10),
        # PredPCA(),
    ]

    # Evaluate models
    results = {model.name: evaluate_model(model, input_train, input_test, label_test) for model in models}

    return results


def evaluate_model(
    model: BaseAutoEncoder,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a single model using specified classifier and metrics

    Args:
        model: Model instance to evaluate
        X_train: Training data (n_features, n_samples)
        X_test: Test data (n_features, n_samples)
        y_test: Test labels (n_samples,)
        classifier: Classifier to use for evaluation. If None, uses ICAWTAClassifier

    Returns:
        Dictionary of metric names to values
    """
    ica = ICA(n_classes=10)
    classifier = WTAClassifier()

    # Train model and classifier
    model.fit(X_train)
    train_encodings = model.encode(X_train)
    ica.fit(train_encodings)

    # Encode and classify test data
    test_encodings = model.encode(X_test)
    test_ica = ica.transform(test_encodings)

    # Compute metrics
    categorization_error = classifier.compute_categorization_error(test_ica, y_test)
    metrics = {
        "categorization_error": categorization_error,
    }

    # Visualize reconstructions
    reconst_data = model.decode(test_encodings)
    visualize_reconstructions(X_test, reconst_data, out_dir / f"{model.name.lower()}_reconstructions.png")

    return metrics


def visualize_reconstructions(
    input_data: np.ndarray,
    reconst_data: np.ndarray,
    filename: str,
):
    import torch
    from torchvision.utils import save_image

    n_samples = 10
    comparison = np.concatenate(
        [
            input_data[:n_samples].reshape(-1, 1, 28, 28),
            reconst_data[:n_samples].reshape(-1, 1, 28, 28),
        ],
        axis=0,
    )
    save_image(torch.from_numpy(comparison), filename, nrow=n_samples)


if __name__ == "__main__":
    main()
