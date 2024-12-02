from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from predpca.mnist.create_digit_sequence import create_digit_sequence
from predpca.models.base_encoder import BaseEncoder
from predpca.models.baselines.vae.encoder import VAE
from predpca.models.baselines.vae.model import VAEModel
from predpca.models.decoder import Decoder
from predpca.models.ica import ICA
from predpca.models.predpca.encoder import PredPCAEncoder
from predpca.models.predpca.model import PredPCA
from predpca.models.wta_classifier import WTAClassifier

sequence_type = 1
prior_u = 1.0

mnist_dir = Path(__file__).parent
data_dir = mnist_dir / "data"
out_dir = mnist_dir / "output" / "model_comparison" / f"sequence_type_{sequence_type}"

np.random.seed(1000000)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    results = compare_models()

    # Display the results
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")


def compare_models(
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
    input_train = input_train.T  # (n_samples, input_dim)
    input_test = input_test.T  # (n_samples, input_dim)
    label_test = label_test.ravel()  # (n_samples,)

    # Prepare encoders
    encoders = [
        VAE(
            model=VAEModel(
                input_dim=784,
                hidden_dim=400,
                latent_dim=10,
            ),
            epochs=10,
        ),
        PredPCAEncoder(
            model=PredPCA(
                kp_list=range(1, 41),
                prior_s_=100,
            ),
            Ns=40,
            Nu=10,
        ),
    ]

    # Evaluate encoders
    results = {encoder.name: evaluate_encoder(encoder, input_train, input_test, label_test) for encoder in encoders}

    return results


def evaluate_encoder(
    encoder: BaseEncoder,
    input_train: np.ndarray,
    input_test: np.ndarray,
    label_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a single encoder using specified classifier and metrics

    Args:
        encoder: Model instance to evaluate
        input_train: Training data (n_samples, input_dim)
        input_test: Test data (n_samples, input_dim)
        label_test: Test labels (n_samples,)
        classifier: Classifier to use for evaluation. If None, uses ICAWTAClassifier

    Returns:
        Dictionary of metric names to values
    """
    input_mean = input_train.mean(axis=0, keepdims=True)  # (1, input_dim)
    input_train_centered = input_train - input_mean
    input_test_centered = input_test - input_mean

    # encode
    encoder.fit(input_train_centered)
    train_encodings = encoder.encode(input_train_centered)
    test_encodings = encoder.encode(input_test_centered)

    # ICA
    ica = ICA(n_classes=10)
    train_ica = ica.fit_transform(train_encodings, test_encodings)
    test_ica = ica.transform(test_encodings)

    # Classify
    classifier = WTAClassifier()
    pred_onehot = classifier.predict(test_ica)

    # Compute metrics
    categorization_error = classifier.compute_categorization_error(pred_onehot, label_test)
    metrics = {
        "categorization_error": categorization_error,
    }

    # Visualize reconstructed images
    if hasattr(encoder, "decode"):
        reconst_images = encoder.decode(test_encodings) + input_mean
        visualize_decodings(input_test, reconst_images, out_dir / f"{encoder.name.lower()}_decodings.png")

    return metrics


def visualize_decodings(
    input_data: np.ndarray,
    reconst_data: np.ndarray,
    filename: str,
):
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
