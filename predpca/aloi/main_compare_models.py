import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from predpca.aloi.predpca_utils import prepare_train_test_sequences
from predpca.models import BaseEncoder, PredPCA, PredPCAEncoder
from predpca.models.baselines import AE, LTAE, TAE, TICA, VAE, AEModel, LTAEModel, PredAE, SimpleNN, TAEModel, VAEModel

aloi_dir = Path(__file__).parent
preproc_out_dir = aloi_dir / "output"
out_dir = aloi_dir / "output" / "model_comparison"

t_train = 57600  # number of training data
t_test = 14400  # number of test data
kf = 12  # future timepoint to be used for prediction targets

Ns = 300  # dimensionality of inputs
Nu = 150  # dimensionality of encoders


def main(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out_subdir = out_dir / f"seed_{seed}"
    out_subdir.mkdir(parents=True, exist_ok=True)
    results = compare_models(t_train, t_test, kf, out_subdir)

    # save results
    with open(out_subdir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Display the results
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")


def compare_models(
    t_train: int,
    t_test: int,
    kf: int,
    out_dir: Path,
):
    s_train, s_test, s_target_train, s_target_test = prepare_data(preproc_out_dir, t_train, t_test, kf)
    s_train_flat = s_train.reshape(Ns, -1).T  # (t_train, Ns)
    s_test_flat = s_test.reshape(Ns, -1).T  # (t_test, Ns)

    # Prepare encoders
    encoders = [
        PredPCAEncoder(
            model=PredPCA(
                kp_list=range(0, 37, 2),
                prior_s_=100,
            ),
            enable_preprocess=False,
            enable_postprocess=False,
        ),
        TAE(
            model=TAEModel(units=[Ns, 250, 200, Nu]),
            batch_size=128,
            epochs=10,
        ),
        TICA(
            dim=Nu,
        ),
        LTAE(
            model=LTAEModel(n_components=Nu),
        ),
        PredAE(
            base_ae=AE(
                model=AEModel(units=[Ns, 250, 200, Nu]),
                batch_size=128,
                epochs=10,
                lr=1e-3,
            ),
            predictor_model=SimpleNN(latent_dim=Nu, hidden_dim=250),
            predictor_epochs=10,
            batch_size=128,
            predictor_lr=1e-3,
        ),
        PredAE(
            base_ae=VAE(
                model=VAEModel(units=[Ns, 250, 200, Nu]),
                batch_size=128,
                epochs=10,
                lr=1e-3,
            ),
            predictor_model=SimpleNN(latent_dim=Nu, hidden_dim=250),
            predictor_epochs=10,
            batch_size=128,
            predictor_lr=1e-3,
        ),
    ]

    # Evaluate encoders
    results = {}
    for encoder in encoders:
        if encoder.name == "PredPCA":
            results[encoder.name] = evaluate_encoder(encoder, s_train, s_test, s_target_train, s_target_test, out_dir)
        else:
            results[encoder.name] = evaluate_encoder(
                encoder, s_train_flat, s_test_flat, s_target_train, s_target_test, out_dir
            )

    return results


def prepare_data(
    preproc_out_dir: Path,
    t_train: int,
    t_test: int,
    kf: int,
    with_noise: bool = False,
):
    npz = np.load(preproc_out_dir / "aloi_preprocessed.npz")
    data = npz["data"][:Ns, :].astype(float)
    # s: (Ns, n_seq, seq_len)
    s_train, s_test, s_target_train, s_target_test = prepare_train_test_sequences(
        data, t_train, t_test, [kf], with_noise
    )
    s_target_train = s_target_train.squeeze().T  # (t_train, Ns)
    s_target_test = s_target_test.squeeze().T  # (t_test, Ns)

    return s_train, s_test, s_target_train, s_target_test


def evaluate_encoder(
    encoder: BaseEncoder,
    s_train: np.ndarray,
    s_test: np.ndarray,
    s_target_train: np.ndarray,
    s_target_test: np.ndarray,
    out_dir: Path,
) -> dict[str, float]:
    """Evaluate a single encoder using specified classifier and metrics

    Args:
        encoder: Model instance to evaluate
        s_train: Training data (n_samples, input_dim)
        s_test: Test data (n_samples, input_dim)
        s_target_train: Training target data (n_samples, input_dim)
        s_target_test: Test target data (n_samples, input_dim)

    Returns:
        Dictionary of metric names to values
    """
    # encode
    encoder.fit(s_train, s_target_train)
    test_pred = encoder.reconstruct(s_test)

    # Compute metrics
    diff = s_target_test - test_pred
    prediction_error = np.mean(np.square(diff)) / np.mean(np.square(s_target_test))
    metrics = {
        "prediction_error": prediction_error,
    }

    # Plot learning curves
    if hasattr(encoder, "train_losses"):
        train_steps, train_losses = encoder.train_losses
        fig = plot_losses(train_steps, train_losses)
        fig.savefig(out_dir / f"{encoder.name.lower()}_losses.png")
        plt.close(fig)

    return metrics


def visualize_decodings(
    input_data: np.ndarray,
    reconst_data: np.ndarray,
    filename: Path,
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


def plot_losses(
    train_steps: np.ndarray,
    train_losses: np.ndarray,
    val_steps: np.ndarray | None = None,
    val_losses: np.ndarray | None = None,
):
    fig = plt.figure()
    plt.plot(train_steps, train_losses, label="Training Loss")
    if val_steps is not None and val_losses is not None:
        plt.plot(val_steps, val_losses, label="Validation Loss", marker="o")
    plt.legend()
    plt.xlabel("Step")
    return fig


if __name__ == "__main__":
    main(seed=1)

    # for seed in range(10):
    #     main(seed=1000000 + seed)
