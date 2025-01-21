import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from predpca.mnist.create_digit_sequence import create_digit_sequence
from predpca.models import ICA, BaseEncoder, PredPCA, PredPCAEncoder, WTAClassifier
from predpca.models.baselines import AE, LTAE, TAE, TICA, VAE, AEModel, LTAEModel, TAEModel, VAEModel

sequence_type = 1
t_train = 100000
t_test = 100000
t_val = 10000

mnist_dir = Path(__file__).parent
data_dir = mnist_dir / "data"
out_dir = mnist_dir / "output" / "model_comparison"


def main(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out_subdir = out_dir / f"sequence_type_{sequence_type}" / f"seed_{seed}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    results = compare_models(t_train, t_test, t_val, out_subdir)

    # save results
    with open(out_subdir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Display the results
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")


def compare_models(t_train: int, t_test: int, t_val: int, out_dir: Path):
    input_train, input_test, input_val, label_test, input_mean = prepare_data(t_train, t_test, t_val)
    is_2step = sequence_type == 2

    # Prepare encoders
    input_dim = input_train.shape[1]
    encoders_nolag_target = [
        VAE(
            model=VAEModel(units=[input_dim, 200, 100, 10]),
            epochs=10,
        ),
        AE(
            model=AEModel(units=[input_dim, 200, 100, 10]),
            epochs=10,
        ),
        PredPCAEncoder(
            model=PredPCA(
                kp_list=range(1, 41),
                prior_s_=100.0,
            ),
            Ns=40,
            Nu=10,
        ),
    ]
    input_dim = input_train.shape[1] * 2 if is_2step else input_train.shape[1]
    encoders_lagged_target = [
        TAE(
            model=TAEModel(units=[input_dim, 200, 100, 10]),
            epochs=10,
        ),
        TICA(
            dim=10,
        ),
        LTAE(
            model=LTAEModel(n_components=10),
        ),
    ]

    # Evaluate encoders
    results = {}

    # encoders with no lagged target
    target_train = input_train
    target_val = input_val
    for encoder in encoders_nolag_target:
        results[encoder.name] = evaluate_encoder(
            encoder, input_train, input_test, input_val, target_train, target_val, label_test, input_mean, out_dir
        )

    # encoders with lagged target
    if is_2step:
        input_train = create_2step_data(input_train)
        input_test = create_2step_data(input_test)
        input_val = create_2step_data(input_val)
        input_mean = create_2step_data(input_mean)

    target_train = np.roll(input_train, -1, axis=0)
    target_val = np.roll(input_val, -1, axis=0)
    for encoder in encoders_lagged_target:
        results[encoder.name] = evaluate_encoder(
            encoder, input_train, input_test, input_val, target_train, target_val, label_test, input_mean, out_dir
        )

    return results


def prepare_data(t_train: int, t_test: int, t_val: int):
    input_train, input_test, input_val, _, label_test, _ = create_digit_sequence(
        data_dir,
        sequence_type,
        t_train,
        t_test,
        t_val,
        train_randomness=True,
        test_randomness=False,
        train_signflip=True,
        test_signflip=False,
    )
    input_train = input_train.T  # (n_samples, input_dim)
    input_test = input_test.T  # (n_samples, input_dim)
    input_val = input_val.T  # (n_samples, input_dim)
    label_test = label_test.ravel()  # (n_samples,)

    # Center the data
    input_mean = input_train.mean(axis=0, keepdims=True)  # (1, input_dim)
    input_train = input_train - input_mean
    input_test = input_test - input_mean
    input_val = input_val - input_mean

    return input_train, input_test, input_val, label_test, input_mean


def create_2step_data(data: np.ndarray) -> np.ndarray:
    """Create input data that includes both current and previous timesteps.

    Args:
        data: Input data of shape (n_samples, input_dim)

    Returns:
        Combined data of shape (n_samples, 2*input_dim) containing current and previous timesteps
    """
    prev_data = np.roll(data, 1, axis=0)
    return np.concatenate([data, prev_data], axis=1)


def extract_current_step_data(data: np.ndarray) -> np.ndarray:
    """Extract current timestep data from combined two-step data.

    Args:
        data: Combined data of shape (n_samples, 2*input_dim)

    Returns:
        Current timestep data of shape (n_samples, input_dim)
    """
    input_dim = data.shape[1] // 2
    return data[:, :input_dim]


def evaluate_encoder(
    encoder: BaseEncoder,
    input_train: np.ndarray,
    input_test: np.ndarray,
    input_val: np.ndarray,
    target_train: np.ndarray,
    target_val: np.ndarray,
    label_test: np.ndarray,
    input_mean: np.ndarray,
    out_dir: Path,
) -> dict[str, float]:
    """Evaluate a single encoder using specified classifier and metrics

    Args:
        encoder: Model instance to evaluate
        input_train: Training data (n_samples, input_dim)
        input_test: Test data (n_samples, input_dim)
        input_val: Validation data (n_samples, input_dim)
        target_train: Training target data (n_samples, input_dim)
        target_val: Validation target data (n_samples, input_dim)
        label_test: Test labels (n_samples,)
        input_mean: Mean of input data for reconstruction (1, input_dim)
        out_dir: Output directory for saving results

    Returns:
        Dictionary of metric names to values
    """
    # encode
    encoder.fit(input_train, target_train, input_val, target_val)
    train_encodings = encoder.encode(input_train)
    test_encodings = encoder.encode(input_test)

    # ICA
    ica = ICA(n_classes=10)
    ica.fit(train_encodings, test_encodings)
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

    # Plot learning curves
    if hasattr(encoder, "train_losses") and hasattr(encoder, "val_losses"):
        train_steps, train_losses = encoder.train_losses
        val_steps, val_losses = encoder.val_losses
        fig = plot_losses(train_steps, train_losses, val_steps, val_losses)
        fig.savefig(out_dir / f"{encoder.name.lower()}_losses.png")
        plt.close(fig)

    return metrics


def visualize_decodings(
    input_data: np.ndarray,
    reconst_data: np.ndarray,
    filename: Path,
):
    n_samples = 10

    # For sequence_type 2, extract current timestep data
    if sequence_type == 2:
        input_data = extract_current_step_data(input_data)
        reconst_data = extract_current_step_data(reconst_data)

    comparison = np.concatenate(
        [
            input_data[:n_samples].reshape(-1, 1, 28, 28),
            reconst_data[:n_samples].reshape(-1, 1, 28, 28),
        ],
        axis=0,
    )
    save_image(torch.from_numpy(comparison), filename, nrow=n_samples)


def plot_losses(
    train_steps: np.ndarray,
    train_losses: np.ndarray,
    val_steps: np.ndarray,
    val_losses: np.ndarray,
):
    fig = plt.figure()
    plt.plot(train_steps, train_losses, label="Training Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss", marker="o")
    plt.legend()
    plt.xlabel("Step")
    return fig


if __name__ == "__main__":
    main(seed=1)

    # for seed in range(10):
    #     main(seed=1000000 + seed)
