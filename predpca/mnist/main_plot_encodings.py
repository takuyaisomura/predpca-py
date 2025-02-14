import csv
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from predpca.mnist.create_digit_sequence import create_digit_sequence
from predpca.mnist.postprocess import postprocessing
from predpca.mnist.visualize import visualize_encodings
from predpca.models import PredPCA

train_randomness = True
train_signflip = True
test_randomness = False
test_signflip = False
T_train = 100000  # training sample size
T_test = 100000  # test sample size

# magnitude of regularization term
prior_s_ = 100.0
prior_u = 1.0

Ns = 40  # input dimensionality
Nu = 10  # encoding dimensionality
Kp = 40  # order of past observations used for prediction

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output" / Path(__file__).stem


def main(
    sequence_type: int,  # 1=ascending, 2=Fibonacci
    data_dir: Path,
    out_dir: Path,
    seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)  # set seed for reproducibility

    # Create input sequences
    print("read files")
    input_train, input_test, _, _, label_test, _ = create_digit_sequence(
        data_dir,
        sequence_type,
        T_train,
        T_test,
        T_test,
        train_randomness,
        test_randomness,
        train_signflip,
        test_signflip,
    )  # input: (784, T), label: (1, T)
    input_mean = input_train.mean(axis=1, keepdims=True)
    input_train -= input_mean
    input_test -= input_mean

    print("compress data using PCA as preprocessing")
    pca_pre = PCA(n_components=Ns)
    s_train = pca_pre.fit_transform(input_train.T).T  # (Ns, T_train)
    s_test = pca_pre.transform(input_test.T).T  # (Ns, T_test)

    # PredPCA
    print("PredPCA")
    predpca = PredPCA(kp_list=range(1, Kp + 1), prior_s_=prior_s_)
    se_train = predpca.fit_transform(s_train, s_train)  # (Ns, T_train)
    se_test = predpca.transform(s_test)  # (Ns, T_test)

    print("- post-hoc PCA using eigenvalue decomposition")
    pca_post = PCA(n_components=Nu)
    u_train = pca_post.fit_transform(se_train.T).T  # (Nu, T_train)
    u_test = pca_post.transform(se_test.T).T  # (Nu, T_test)

    # ICA
    print("ICA")
    ui_train, ui_test, Wica, v_test, conf_mat = postprocessing(u_train, u_test, label_test)
    # ui_train: (Nu, T_train), ui_test: (Nu, T_test), v_test: (Nv, T_test), conf_mat: (Nv, Nv)

    # Sort latent variables based on confusion matrix
    latent_to_digit = conf_mat.argmax(axis=0)  # For each latent variable, find the most corresponding digit
    digit_to_latent = np.argsort(latent_to_digit)  # Reorder latents to match digits 0-9
    ui_test_sorted = ui_test[digit_to_latent]

    # Plotting
    fig = visualize_encodings(ui_test_sorted, label_test, range(T_test // 10))
    fig.savefig(out_dir / "encodings.png")

    # Output files for Fig 2a
    if sequence_type == 1:
        # mapping from encoding states to digit images
        A = (
            input_train @ ui_train.T @ linalg.inv(ui_train @ ui_train.T + np.eye(Nu) * prior_u)
        )  # (784, Nu): (784, T) @ (T, Nu) @ (Nu, Nu)
        ui_mean = np.zeros((Nu, Nu))
        for i in range(Nu):
            ui_mean[:, i] = ui_test[:, v_test[i, :] == 1].mean(axis=1)
        mean_images = A @ ui_mean + input_mean  # (784, Nu)

        # save categorization results
        with open(out_dir / "encodings.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(range(10)) + [-1])
            writer.writerows(np.concatenate((ui_test, label_test), axis=0).T)

        with open(out_dir / "input.dat", "w") as f:
            np.uint8((input_test[:, :1000] + input_mean) * 255).tofile(f)

        with open(out_dir / "estimated_input.dat", "w") as f:
            W_pca_pre = pca_pre.components_  # (Ns, 784)
            np.uint8((W_pca_pre.T @ se_test[:, :1000] + input_mean) * 255 * 1.2).tofile(f)

        with open(out_dir / "mean_images.dat", "w") as f:
            np.uint8(mean_images * 255 * 1.2).tofile(f)

        W_pca_post = pca_post.components_
        return W_pca_pre, W_pca_post, Wica, A


if __name__ == "__main__":
    main(
        sequence_type=1,
        data_dir=data_dir,
        out_dir=out_dir,
    )
