import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from predpca.aloi.predpca_utils import predict_encoding, prediction_error, prepare_train_test_sequences
from predpca.aloi.visualize import plot_hidden_state, plot_test_images, plot_true_and_pred_video
from predpca.models import PredPCA

start_time = time.time()

T_train = 57600  # number of training data
T_test = 14400  # number of test data

Ns = 300  # dimensionality of inputs
Nu = 128  # dimensionality of encoders
Nv = 20  # number of hidden states to visualize

Kp_list = range(0, 37, 2)  # past timepoints to be used for basis functions
Kp2_list = range(37)  # past timepoints to be used for basis functions (optimal)
Kf_list = [6, 12, 18, 24, 30]  # future timepoints to be used for prediction targets
Kf_viz = 2  # Kf_viz-th timepoint in Kf_list will be visualized
Kp = len(Kp_list)
Kf = len(Kf_list)
WithNoise = False  # presence of noise

# Priors (small constants) to prevent inverse matrix from being singular
prior_x = 1.0
prior_s = 100.0
prior_s_ = 100.0
prior_so_ = 100.0
prior_qSigmao = 0.01

# limited number of training samples
NT = 8  # number of sections
Nu_list = Nu // NT * np.arange(1, NT + 1)  # dimensionality of encoders

out_dir = Path(__file__).parent / "output" / Path(__file__).stem


def main(
    out_dir: Path,
    preproc_out_dir: Path = Path(__file__).parent / "output",
    seed: int = 0,
    visualize: bool = True,
):
    print("----------------------------------------")
    print("PredPCA of 3D rotating object images")
    print("----------------------------------------\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)

    print("loading data")
    npz = np.load(preproc_out_dir / "aloi_preprocessed.npz")
    data = npz["data"][:Ns, :].astype(float)  # (Ns, Timg)

    print("preprocessing")
    s_train, s_test, s_target_train, s_target_test = prepare_train_test_sequences(
        data, T_train, T_test, Kf_list, WithNoise
    )

    print("investigate the accuracy with limited number of training samples")

    err_s1 = {
        "em": np.zeros((Kf, Ns, NT)),  # PredPCA empirical
        "em_opt": np.zeros((Kf, Ns, NT)),  # PredPCA empirical (optimal)
    }

    _, n_seq_train, seq_len = s_train.shape
    for h in range(NT):
        n_seq_sub = n_seq_train * (h + 1) // NT  # number of training sequences
        T_sub = n_seq_sub * seq_len  # number of training samples
        print(f"training with T = {T_sub} (time = {(time.time() - start_time) / 60:.1f} min)")

        s_sub = s_train[:, :n_seq_sub]
        s_target_sub = s_target_train[:, :, :T_sub]

        print("compute PredPCA with plain bases")
        predpca = PredPCA(kp_list=Kp_list, prior_s_=prior_s_)
        pred_err_em, se_sub, _, W_pca_post = run_with_limited_samples(
            predpca,
            s_sub,
            s_test,
            s_target_sub,
            s_target_test,
            Nu_list[h],
        )
        err_s1["em"][:, :, h] = pred_err_em

        qA, qSigmao = identify_system_param(W_pca_post, s_sub, se_sub)

        print("compute PredPCA with optimal bases")
        # optimal basis functions
        gain = qA.T @ linalg.inv(qSigmao)  # optimal gain; (Nu, Ns)
        predpca_opt = PredPCA(kp_list=Kp2_list, prior_s_=prior_so_, gain=gain)
        pred_err_em_opt, se_sub, se_test, W_pca_post_opt = run_with_limited_samples(
            predpca_opt,
            s_sub,
            s_test,
            s_target_sub,
            s_target_test,
            Nu_list[h],
        )
        err_s1["em_opt"][:, :, h] = pred_err_em_opt

    print(f"search complete (time = {(time.time() - start_time) / 60:.1f} min)")
    print("----------------------------------------\n")

    # postprocessing
    print("postprocessing")
    u_sub, _, du_sub, du_test = predict_encoding(W_pca_post_opt, se_sub, se_test)

    # postprocess error
    norm_s_test = (s_target_test[0] ** 2).sum(axis=0).mean()
    err_s1["em"] /= norm_s_test
    err_s1["em_opt"] /= norm_s_test
    err_dst = np.zeros((Kf, NT))
    for h in range(NT):
        err_dst[:, h] = err_s1["em_opt"][:, Nu_list[h] - 1, h]

    # test prediction error (for Fig 3e)
    save_error(err_dst, err_s1, out_dir)

    print("----------------------------------------\n")

    if not visualize:
        return

    u_test_viz = W_pca_post_opt @ se_test[Kf_viz]  # predictive encoders (test) (Nu[h], T_test)

    # plot test prediction error
    plot_test_pred_error(err_dst, err_s1["em"], out_dir)

    PCA_C1 = npz["PCA_C1"]  # (2, 2, Ndata1, Ndata1)
    PCA_C2 = npz["PCA_C2"]  # (Ndata2, Ndata2)
    mean1 = npz["mean1"]  # (2, 2, Ndata1)

    # true and predicted images (for Fig 3a and Suppl Movie)
    print(f"true and predicted images (time = {(time.time() - start_time) / 60:.1f} min)")
    print("create supplementary movie")
    plot_true_and_pred_video(W_pca_post_opt, u_test_viz, s_target_test, PCA_C1, PCA_C2, mean1, out_dir)
    print("----------------------------------------\n")

    # hidden state analysis
    # ICA of mean encoders (for Fig 3b and Suppl Fig 4)
    print(f"ICA of mean encoders (time = {(time.time() - start_time) / 60:.1f} min)")
    plot_hidden_state(W_pca_post, u_sub, PCA_C1, PCA_C2, mean1, Nv, WithNoise, out_dir)

    # PCA of deviation encoders (for Fig 3c)
    print(f"PCA of deviation encoders (time = {(time.time() - start_time) / 60:.1f} min)")
    plot_deviation_encoder(du_sub, du_test, out_dir)

    # plot test images
    plot_test_images(W_pca_post_opt, u_test_viz, PCA_C1, PCA_C2, mean1, out_dir)

    print("----------------------------------------\n")


def run_with_limited_samples(
    predpca,
    s_sub,  # (Ns, n_seq, seq_len)
    s_test,  # (Ns, n_seq, seq_len)
    s_target_sub,  # (Kf, Ns, T_sub)
    s_target_test,  # (Kf, Ns, T_test)
    Nu,  # int
):
    se_sub = predpca.fit_transform(s_sub, s_target_sub)  # (Kf, Ns, T_sub)
    se_test = predpca.transform(s_test)  # (Kf, Ns, T_test)

    C = predpca.predict_input_pca(se_sub)[0]  # (Ns, Ns)
    W_pca_post_opt = C[:, :Nu].T  # optimal weights = transpose of eigenvectors (Nu, Ns)

    # test prediction error
    pred_err_em_opt = prediction_error(s_target_test, se_test, C)  # (Kf, Ns)

    return pred_err_em_opt, se_sub, se_test, W_pca_post_opt


def identify_system_param(
    W_pca_post,  # (Nu, Ns)
    s_sub,  # (Ns, n_seq, seq_len)
    se_sub,  # (Kf, Ns, T)
):
    T = se_sub.shape[-1]
    Nu, Ns = W_pca_post.shape

    s_sub = s_sub.reshape(Ns, -1)  # flatten; (Ns, T)

    # system parameter identification
    u_sub = W_pca_post @ se_sub[0]  # predictive encoders (training) (Nu, T)
    uc_sub = W_pca_post @ s_sub  # current input encoders (training) (Nu, T)
    qA = W_pca_post.T  # observation matrix (Ns, Nu)
    qPsi = (np.roll(u_sub, -1, axis=1) @ u_sub.T) @ linalg.inv(
        u_sub @ u_sub.T + np.eye(Nu) * T / T_train
    )  # transition matrix (Nu, Nu)
    qSigmas = (s_sub @ s_sub.T) / T  # input covariance (Ns, Ns)
    qSigmap = linalg.inv(qPsi) @ (np.roll(uc_sub, -1, axis=1) @ uc_sub.T / T)  # (Nu, Nu)
    qSigmap = (qSigmap + qSigmap.T) / 2  # hidden basis covariance (Nu, Nu)
    qSigmao = qSigmas - qA @ qSigmap @ qA.T  # observation noise covariance (Ns, Ns)
    U, S, _ = linalg.svd(qSigmao)
    U = np.array(U)  # (Ns, Ns)
    S = np.maximum(np.diag(S), np.eye(Ns) * prior_qSigmao)  # make all eigenvalues positive; (Ns, Ns)
    qSigmao = U @ S @ U.T  # correction; (Ns, Ns)

    return qA, qSigmao


def save_error(
    err_dst,
    err_s1,
    out_dir,
):
    # save test prediction error
    output_data = np.vstack(
        [
            np.tile(np.arange(1, 3), Kf),
            np.hstack([np.column_stack((row1, row2)) for row1, row2 in zip(err_dst, err_s1["em"][:, -1, :])]),
        ]
    )
    np.savetxt(out_dir / "predpca_test_err.csv", output_data, delimiter=",")

    # save optimal encoding dimensionality (for Fig 3d)
    print(f"optimal encoding dimensionality (time = {(time.time() - start_time) / 60:.1f} min)")
    err_mean = err_s1["em"].mean(axis=0)  # (Ns, NT)
    idx = err_mean.argmin(axis=0)  # (NT,)
    output_data = np.vstack([np.arange(1, NT + 1), idx])
    np.savetxt(out_dir / "predpca_opt_encode_dim.csv", output_data, delimiter=",", fmt="%d")


def plot_test_pred_error(err_dst, err_em, out_dir):
    fig, axs = plt.subplots(3, 2, figsize=(8, 10))
    for k in range(Kf):
        ax = axs[k // 2, k % 2]
        ax.plot(np.arange(2, NT + 1) * (800 / NT), err_dst[k, 1:], "-b")
        ax.plot(np.arange(2, NT + 1) * (800 / NT), err_em[k, -1, 1:], "--b")
        ax.set_xlim(2 * (800 / NT), NT * (800 / NT))
        ax.set_ylim(0.0, 1.2)
        ax.set_title(f"test error ({(k + 1) * 30} deg rot)")
    axs[-1, -1].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "predpca_test_err.png")


def plot_deviation_encoder(
    du_sub,
    du_test,
    out_dir,
):
    pca = PCA().fit(du_sub[Kf_viz, :Nv, :].T)
    C = pca.components_  # (Nv, Nv)
    L = pca.explained_variance_  # (Nv,)
    pc1 = C[0] @ du_test[Kf_viz, :Nv, :]  # (Nv, T_test)
    pc1 /= pc1.std(axis=0, ddof=1)

    plt.figure(figsize=(10, 6))
    pc1_ = pc1.reshape(72, -1, order="F")
    plt.plot(pc1_, "c-")
    plt.axhline(y=0, color="k", linestyle="--", linewidth=3)
    plt.plot(np.quantile(pc1_.T, 0.2, axis=0), "k-", linewidth=3)
    plt.plot(np.quantile(pc1_.T, 0.5, axis=0), "k-", linewidth=3)
    plt.plot(np.quantile(pc1_.T, 0.8, axis=0), "k-", linewidth=3)
    plt.xlim(1, 72)
    plt.ylim(-2, 2)
    plt.savefig(out_dir / "predpca_pc1_of_deviation.png")
    plt.close()

    np.savetxt(
        out_dir / "predpca_pc1_of_deviation.csv",
        np.vstack(
            [
                np.arange(1, T_test // 72 + 1),
                pc1_,
            ]
        ),
        delimiter=",",
    )
    np.savetxt(
        out_dir / "predpca_pc1_of_deviation_eig.csv",
        np.vstack(
            [
                np.arange(1, 3),
                np.column_stack([L, L / np.sum(L)]),
            ]
        ),
        delimiter=",",
    )


if __name__ == "__main__":
    main(
        out_dir=out_dir,
        seed=0,
        visualize=True,
    )
