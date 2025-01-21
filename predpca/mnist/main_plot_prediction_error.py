import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from predpca.mnist.create_digit_sequence import create_digit_sequence
from predpca.mnist.predpca_utils import (
    calculate_estimated_parameters,
    calculate_true_parameters,
    generate_true_hidden_states,
)
from predpca.models import PredPCA

train_randomness = True
test_randomness = True
train_signflip = True
test_signflip = True
T_train = 100000  # training sample size
T_test = 100000  # test sample size
T_val = 100000  # data size used for determining true parameters

prior_x = 1.0  # magnitude of regularization term
prior_s_ = 100.0

Ns = 40
Kp = 10
Nx = 10
Nu = 10
NT = 19

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output" / Path(__file__).stem


def main(
    sequence_type: int,  # 1=ascending, 2=Fibonacci
    data_dir: Path,
    out_dir: Path,
    seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)

    # create input sequences
    print("read files")
    input_train, input_test, input_val, label_train, label_test, label_val = create_digit_sequence(
        data_dir,
        sequence_type,
        T_train,
        T_test,
        T_val,
        train_randomness,
        test_randomness,
        train_signflip,
        test_signflip,
    )  # input: (784, T), label: (1, T)
    input_mean = input_train.mean(axis=1, keepdims=True)  # (784, 1)
    input_train = input_train - input_mean
    input_test = input_test - input_mean
    input_val = input_val - input_mean

    print("compress data using PCA as preprocessing")
    pca_pre = PCA(n_components=Ns)
    W_pca_pre = pca_pre.fit(input_train.T).components_  # (Ns, 784)
    s_train = W_pca_pre @ input_train  # (Ns, T_train)
    s_test = W_pca_pre @ input_test  # (Ns, T_test)
    s_val = W_pca_pre @ input_val  # (Ns, T_val)
    # match the variance
    train_std = np.diag(np.std(s_train, axis=1, ddof=1))
    s_test = train_std @ np.diag(1 / np.sqrt(np.mean(s_test**2, axis=1))) @ s_test
    s_val = train_std @ np.diag(1 / np.sqrt(np.mean(s_val**2, axis=1))) @ s_val

    print("compute maximum likelihood estimator")
    print("compute maximum likelihood estimator")
    predpca = PredPCA(kp_list=range(1, Kp + 1), prior_s_=prior_s_)
    se_train = predpca.fit_transform(s_train, s_train)

    # true states and parameters
    print("compute true states and parameters")
    x_train = generate_true_hidden_states(input_train + input_mean, label_train)  # (Nx, T_train)
    x_test = generate_true_hidden_states(input_test + input_mean, label_test)  # (Nx, T_test)
    x_val = generate_true_hidden_states(input_val + input_mean, label_val)  # (Nx, T_val)
    x_mean = x_train.mean(axis=1, keepdims=True)  # (Nx, 1)
    x_train = x_train - x_mean
    x_test = x_test - x_mean
    x_val = x_val - x_mean
    (
        A,  # (Ns, Nx)
        B,  # (Nx, Nx)
        Sigmas,  # (Ns, Ns)
        Sigmax,  # (Nx, Nx)
        Sigmao,  # (Ns, Ns)
        Sigmaz,  # (Nx, Nx)
    ) = calculate_true_parameters(s_val, x_val, prior_x)

    print("optimal state and parameter estimators obtained using supervised learning")
    predpca_opt = PredPCA(kp_list=range(1, Kp + 1), prior_s_=prior_s_)
    qx_opt_train = predpca_opt.fit_transform(s_train, x_train)
    qx_opt_test = predpca_opt.transform(s_test)
    Aopt = compute_A(s_train, x_train, prior_x)

    print("optimal state and parameter estimators obtained using PredPCA")
    qSigmas_opt = s_train @ s_train.T / T_train  # (Ns, Ns)
    Copt, _, qSigmase_opt = predpca.predict_input_pca(se_train)

    err_qSigmase_opt = np.trace(qSigmas_opt - qSigmase_opt)
    err_A_opt1 = Nx * np.trace(qSigmas_opt - Aopt @ (x_train @ x_train.T) @ Aopt.T / T_train)
    err_A_opt2 = Ns * Kp * np.trace(Aopt @ (x_train @ x_train.T - qx_opt_train @ qx_opt_train.T) @ Aopt.T / T_train)

    # prediction errors
    err_optimal = np.zeros(NT)  # optimal
    err_predpca = {
        "th": np.zeros((Ns, NT)),  # PredPCA theory
        "em": np.zeros((Ns, NT)),  # PredPCA empirical
    }
    err_supervised = {
        "th": np.zeros(NT),  # supervised learning theory
        "em": np.zeros(NT),  # supervised learning empirical
    }
    err_param = np.zeros((6, NT))  # parameter estimation error

    print("investigate the accuracy with limited number of training samples")

    for h in range(NT):
        if h < 9:
            T_sub = T_train // 100 * (h + 1)
        else:
            T_sub = T_train // 10 * (h - 8)
        print(f"number of training samples: {T_sub}")
        s_sub = s_train[:, :T_sub]  # (Ns, T_sub)
        x_sub = x_train[:, :T_sub]  # (Nx, T_sub)

        s_sub = s_sub - s_sub.mean(axis=1, keepdims=True)
        x_sub = x_sub - x_sub.mean(axis=1, keepdims=True)

        # PredPCA
        predpca_sub = PredPCA(kp_list=range(1, Kp + 1), prior_s_=prior_s_)
        se_sub = predpca_sub.fit_transform(s_sub, s_sub)  # input expectations (Ns, T_sub)
        se_test = predpca_sub.transform(s_test)  # input expectations (Ns, T_test)

        # eigenvalue decomposition
        pca_post = PCA(n_components=Ns)
        W_pca_post = pca_post.fit(se_sub.T).components_[:Nu]  # eigenvectors (Nu, Ns)
        u_train = W_pca_post @ se_sub  # encoders (training) / prediction (Nu, T_sub)
        uc_train = W_pca_post @ s_sub  # encoders (training) / based on current input (Nu, T_sub)

        # test prediction error
        err_optimal[h] = np.mean(np.sum((s_test - A @ qx_opt_test) ** 2, axis=0))  # optimal

        # supervised learning
        # theory
        err_supervised["th"][h] = err_qSigmase_opt + err_A_opt1 / T_sub + err_A_opt2 / T_sub
        # empirical
        predpca_sl = PredPCA(kp_list=range(1, Kp + 1), prior_s_=prior_s_)
        predpca_sl.fit(s_sub, x_sub)
        qx_sl_test = predpca_sl.transform(s_test)  # hidden state expectation (Nx, T_test)
        A_sl = compute_A(s_sub, x_sub, prior_x)  # mapping (Ns, Nx)
        err_supervised["em"][h] = np.mean(np.sum((s_test - A_sl @ qx_sl_test) ** 2, axis=0))

        # PredPCA
        for i in range(Ns):
            # theory
            Wi = Copt[:, : i + 1].T  # mapping (i + 1, Ns)
            err_predpca["th"][i, h] = (
                np.trace(qSigmas_opt)
                - np.trace(Wi @ qSigmase_opt @ Wi.T)
                + Kp * Ns / T_sub * np.trace(Wi @ (qSigmas_opt - qSigmase_opt) @ Wi.T)
            )
            # empirical
            Wi = pca_post.components_[: i + 1]  # mapping (i + 1, Ns)
            ui_test = Wi @ se_test  # encoder (i + 1, T_test)
            err_predpca["em"][i, h] = np.mean(np.sum((s_test - Wi.T @ ui_test) ** 2, axis=0))

        # system parameter identification
        # ambiguity factor (coordinate transformation)
        Omega_inv = linalg.inv(A.T @ A + np.eye(Nu) * 1e-4) @ A.T @ W_pca_post.T
        # this function is for ascending order
        prior_psi = prior_x * T_sub / T_train
        qA, qB, qSigmas, qSigmax, qSigmao, qSigmaz = calculate_estimated_parameters(
            s_sub, u_train, uc_train, W_pca_post, Omega_inv, prior_psi
        )

        err_param[0, h] = compute_param_error(A, qA)
        err_param[1, h] = compute_param_error(B, qB)
        err_param[2, h] = compute_param_error(Sigmas, qSigmas)
        err_param[3, h] = compute_param_error(Sigmax, qSigmax)
        err_param[4, h] = compute_param_error(Sigmao, qSigmao)
        err_param[5, h] = compute_param_error(Sigmaz, qSigmaz)

    plot_prediction_error(err_param, err_optimal, err_predpca, err_supervised, s_test, out_dir, seed)

    return err_param, err_optimal, err_predpca, err_supervised


def compute_A(s, x, prior_x):
    return (s @ x.T) @ linalg.inv(x @ x.T + np.eye(Nx) * prior_x)


def compute_param_error(param, param_pred):
    max_norm = max(linalg.norm(param, "fro") ** 2, linalg.norm(param_pred, "fro") ** 2)
    return linalg.norm(param - param_pred, "fro") ** 2 / max_norm


def plot_prediction_error(
    err_param,
    err_optimal,
    err_predpca,
    err_supervised,
    s_test,
    out_dir,
    seed,
):
    T_test = s_test.shape[1]

    plt.figure(figsize=(8, 5))

    # for Fig 2c
    # save results
    with open(out_dir / f"err_param_{seed}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(range(1, NT + 1))
        writer.writerows(err_param)

    # show results
    t = np.concatenate([np.arange(1, 11) * 1e3, np.arange(2, 11) * 1e4])

    plt.subplot(1, 2, 1)
    plt.semilogx(t, err_param[0, :] * 100, "k+", label=r"$A$")
    plt.semilogx(t, err_param[1, :] * 100, "r+", label=r"$B$")
    plt.semilogx(t, err_param[2, :] * 100, "g+", label=r"$\Sigma_s$")
    plt.semilogx(t, err_param[3, :] * 100, "b+", label=r"$\Sigma_x$")
    plt.semilogx(t, err_param[4, :] * 100, "y+", label=r"$\Sigma_o$")
    plt.xlim(1e3, 1e5)
    plt.xlabel("Training samples")
    plt.ylabel("Parameter estimation error (%)")
    plt.legend()

    # for Fig 2d
    # save results
    norm_s_test = np.trace(s_test @ s_test.T / T_test)
    with open(out_dir / f"err_pred_{seed}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([1] + list(range(1, Ns + 1)) + list(range(-1, -Ns - 1, -1)) + [1, -1])
        writer.writerows(
            np.concatenate(
                [
                    err_optimal[:, np.newaxis] / norm_s_test,  # (NT, 1)
                    err_predpca["em"].T / norm_s_test,  # (NT, Ns)
                    err_predpca["th"].T / norm_s_test,  # (NT, Ns)
                    err_supervised["em"][:, np.newaxis] / norm_s_test,  # (NT, 1)
                    err_supervised["th"][:, np.newaxis] / norm_s_test,  # (NT, 1)
                ],
                axis=1,
            )
        )

    # show results
    t = np.concatenate([np.arange(1, 11) * 1e3, np.arange(2, 11) * 1e4])
    norm_s_test = np.mean(np.sum(s_test**2, axis=0))

    plt.subplot(1, 2, 2)
    plt.semilogx(t, err_optimal / norm_s_test, "-k", label="Optimal")
    plt.semilogx(t, err_supervised["th"] / norm_s_test, "-b", label="supervised learning (theory)")
    plt.semilogx(t, err_supervised["em"] / norm_s_test, "+b", label="supervised learning (empirical)")
    plt.semilogx(t, err_predpca["th"][Nx - 1, :] / norm_s_test, "-g", label=f"PredPCA ({Nx}-th PC, theory)")
    plt.semilogx(t, err_predpca["em"][Nx - 1, :] / norm_s_test, "+g", label=f"PredPCA ({Nx}-th PC, empirical)")
    plt.semilogx(t, err_predpca["th"][Ns - 1, :] / norm_s_test, "-y", label=f"PredPCA ({Ns}-th PC, theory)")
    plt.semilogx(t, err_predpca["em"][Ns - 1, :] / norm_s_test, "+y", label=f"PredPCA ({Ns}-th PC, empirical)")
    plt.axis([1e3, 1e5, 0.4, 0.8])
    plt.xlabel("Training samples")
    plt.ylabel("Test prediction error (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"prediction_error_seed{seed}.png")
    plt.show()


if __name__ == "__main__":
    main(
        sequence_type=1,
        data_dir=data_dir,
        out_dir=out_dir,
        seed=0,
    )
