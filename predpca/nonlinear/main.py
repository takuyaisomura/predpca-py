from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from predpca.models import PredPCA
from predpca.nonlinear.canonical_nonlinear_system import canonical_nonlinear_system
from predpca.nonlinear.lorenz_attractor import lorenz_attractor
from predpca.utils import pcacov

T_train = 100000  # number of training samples
T_test = 100000  # number of test samples
Npsi = 100  # dimension of hidden basis
Ns = 200  # number of observed variables
sigma_z = 0.001  # noise level of hidden basis
sigma_o = 0.1  # noise level of observed variables

Kp_list = range(1, 11)
prior_s_ = 1.0

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output"


def main(
    sequence_type: int,  # 1 = canonical nonlinear system, 2 = lorenz attractor
    out_dir: Path,
    seed: int = 0,
):
    """Main function to run PredPCA analysis"""
    out_dir /= f"type{sequence_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)

    # Generate data
    # (Nx, T_train), (Nx, T_test), (Npsi, T_train), (Npsi, T_test)
    x_train, x_test, psi_train, psi_test = generate_hidden_state(sequence_type)
    s_train, s_test = generate_input(psi_train, psi_test)  # (Ns, T_train), (Ns, T_test)
    Nx = x_train.shape[0]

    # Perform PredPCA
    predpca = PredPCA(kp_list=Kp_list, prior_s_=prior_s_)
    qs_train = predpca.fit_transform(s_train, s_train)  # (Ns, T_train)
    qs_test = predpca.transform(s_test)  # (Ns, T_test)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=Npsi).fit(qs_train.T)
    Cs = pca.components_  # (Npsi, Ns)
    Ls = pca.explained_variance_  # (Npsi,)
    qpsi_train = Cs @ qs_train  # (Npsi, T_train)
    qpsi_test = Cs @ qs_test  # (Npsi, T_test)
    qpsic = Cs @ s_train  # (Npsi, T_train)

    # Estimate transition matrix
    qPsi = linalg.solve(
        (qpsi_train @ qpsi_train.T).T,
        (np.roll(qpsi_train, -1, axis=1) @ qpsi_train.T).T,
    ).T  # (Npsi, Npsi)

    # Estimate hidden basis covariance
    U, svals, Vh = linalg.svd(qPsi)  # (Npsi, Npsi), (Npsi,), (Npsi, Npsi)
    reg_inv = Vh.T @ np.diag(1 / (svals + 0.001)) @ U.T
    qSigmap = reg_inv @ np.roll(qpsic, -1, axis=1) @ qpsic.T / T_train  # (Npsi, Npsi)
    qSigmap = (qSigmap + qSigmap.T) / 2
    qSigmap += np.eye(Npsi) * 5.0  # avoid singular matrix

    # Compute eigenvalues and components
    Cp, Lp, _ = pcacov(qSigmap)
    pca = PCA().fit(psi_train.T)
    L = pca.explained_variance_  # (Npsi,)

    # Plot eigenvalue analysis
    plot_eigenvalue_analysis(L, Lp, Ls, out_dir)

    # Estimate states
    qx_train, qx_test = estimate_states(qpsi_train, qpsi_test, x_train, Cp, Lp, Nx)

    # Estimate parameters
    B, qB = estimate_parameters(sequence_type, x_train, qx_train, psi_train, qpsi_train, T_train, Nx)

    # Save results and create plots
    save_results(out_dir, seed, x_test, qx_test, Nx, T_test, L, Lp, B, qB)
    plot_states(x_test, qx_test, out_dir)
    amp = 400 if sequence_type == 1 else 2000
    plot_params(B, qB, amp, out_dir)

    return qx_train, qx_test, qpsi_train, qpsi_test, B, qB, Cp, Lp


def estimate_states(qpsi_train, qpsi_test, x_train, Cp, Lp, Nx):
    """Estimate state variables"""
    var_inv = np.diag(1 / np.sqrt(Lp[:Nx]))  # (Nx, Nx)
    qx_train = var_inv @ Cp[:, :Nx].T @ qpsi_train  # (Nx, T_train)
    qx_test = var_inv @ Cp[:, :Nx].T @ qpsi_test  # (Nx, T_test)

    # Compute ambiguity factor that resolves the inherent indeterminacy between the estimated and true state variables
    Omegax = linalg.solve((qx_train @ qx_train.T).T, (x_train @ qx_train.T).T).T  # (Nx, Nx)
    qx_train = Omegax @ qx_train  # (Nx, T_train)
    qx_test = Omegax @ qx_test  # (Nx, T_test)

    return qx_train, qx_test


def generate_hidden_state(sequence_type: int):
    if sequence_type == 1:
        Nx = 10
        x_train, x_test, psi_train, psi_test = canonical_nonlinear_system(Nx, Npsi, T_train, T_test, sigma_z)
    elif sequence_type == 2:
        Nx = 3
        x_train, x_test = lorenz_attractor(Nx, T_train, T_test)
        # generate feature variables
        # add bias term
        x_train_bias = np.vstack([x_train, np.ones((1, T_train))])
        x_test_bias = np.vstack([x_test, np.ones((1, T_test))])
        # create random weights
        R = np.random.randn(Npsi, Nx + 1) / np.sqrt(Nx + 1)
        # apply weights and add nonlinearity
        psi_train = np.tanh(R @ x_train_bias)
        psi_test = np.tanh(R @ x_test_bias)
    else:
        raise ValueError(f"Invalid sequence type: {sequence_type}")

    return x_train, x_test, psi_train, psi_test


def generate_input(psi_train, psi_test):
    # Generate noisy observations
    # Create random observation matrix
    A = np.random.randn(Ns, Npsi)
    # Add observation noise
    omega_train = np.random.randn(Ns, T_train) * sigma_o
    omega_test = np.random.randn(Ns, T_test) * sigma_o
    s_train = A @ psi_train + omega_train
    s_test = A @ psi_test + omega_test
    # Center the data
    s_train -= s_train.mean(axis=1, keepdims=True)
    s_test -= s_test.mean(axis=1, keepdims=True)

    return s_train, s_test


def plot_states(x_test, qx_test, out_dir):
    T0 = 10000
    T1 = 20000
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x_test[0, T0:T1], x_test[1, T0:T1], x_test[2, T0:T1])
    ax1.set_title("Original")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")  # type: ignore
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(qx_test[0, T0:T1], qx_test[1, T0:T1], qx_test[2, T0:T1])
    ax2.set_title("Prediction")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")  # type: ignore
    plt.tight_layout()
    fig.savefig(out_dir / "states.png")

    plt.figure()
    plt.plot(range(T0, T1), x_test[0, T0:T1], label="Original $x_t$")
    plt.plot(range(T0, T1), qx_test[0, T0:T1], label=r"Prediction $\mathbf{x}_{t \mid t-1}$")
    plt.legend()
    plt.xlabel("Time")
    plt.savefig(out_dir / "states_zoom.png")


def plot_params(B, qB, amp, out_dir):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(B * amp))
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(qB * amp))
    plt.tight_layout()
    plt.savefig(out_dir / "param_B.png")


def save_results(out_dir, seed, x_test, qx_test, Nx, T_test, L, Lp, B, qB):
    """Save results to CSV files"""
    np.savetxt(
        out_dir / f"states_{seed}.csv",
        np.vstack(
            [
                np.arange(1, Nx + 1),
                np.arange(1, Nx + 1),
                x_test[:, : T_test // 10].T,
                qx_test[:, : T_test // 10].T,
            ]
        ),  # (2 + T_test // 10 * 2, Nx)
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"correlation_{seed}.csv",
        np.vstack(
            [
                np.arange(1, Nx + 1),
                np.corrcoef(x_test, qx_test).T[:Nx, Nx:],
            ]
        ),  # (1 + Nx, Nx)
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"eigenvalues_{seed}.csv",
        np.vstack(
            [
                np.array([1, 2]),
                np.column_stack([L, Lp]),
            ]
        ),  # (1 + Npsi, 2)
        delimiter=",",
    )
    np.savetxt(
        out_dir / f"param_B_{seed}.csv",
        np.vstack(
            [
                np.arange(1, B.shape[1] + 1),
                B,
                qB,
            ]
        ),  # (1 + B.shape[0] * 2, B.shape[1])
        delimiter=",",
    )


def plot_eigenvalue_analysis(L, Lp, Ls, out_dir):
    """Plot eigenvalue analysis results"""
    idx = np.arange(1, Npsi + 1)

    plt.figure()
    # eigenvalues
    plt.subplot(1, 2, 1)
    plt.plot(idx, L)
    plt.plot(idx, Lp)
    plt.plot(idx, Ls)
    # diff of eigenvalues
    plt.subplot(1, 2, 2)
    plt.plot(idx, L - np.append(L[1:], 0))
    plt.plot(idx, Lp - np.append(Lp[1:], 0))
    plt.savefig(out_dir / "eigenvalues.png")
    plt.close()


def estimate_parameters(sequence_type, x_train, qx_train, psi_train, qpsi_train, T_train, Nx):
    """Estimate system parameters based on sequence type"""
    if sequence_type == 1:
        B = linalg.solve(
            (psi_train @ psi_train.T + np.eye(Npsi)).T,  # (Npsi, Npsi)
            (np.roll(x_train, -1, axis=1) @ psi_train.T).T,  # (Npsi, Nx)
        ).T  # (Nx, Npsi)
        qB = linalg.solve(
            (qpsi_train @ qpsi_train.T + np.eye(Npsi)).T,  # (Npsi, Npsi)
            (np.roll(qx_train, -1, axis=1) @ qpsi_train.T).T,  # (Npsi, Nx)
        ).T  # (Nx, Npsi)

    elif sequence_type == 2:
        xx = np.vstack(
            [
                np.ones(T_train),  # (1, T_train)
                x_train,  # (Nx, T_train)
                x_train**2,  # (Nx, T_train)
                x_train[1] * x_train[2],  # (1, T_train)
                x_train[2] * x_train[0],  # (1, T_train)
                x_train[0] * x_train[1],  # (1, T_train)
            ]
        )  # (1 + Nx * 2 + 3, T_train)

        qxx = np.vstack(
            [
                np.ones(T_train),  # (1, T_train)
                qx_train,  # (Nx, T_train)
                qx_train**2,  # (Nx, T_train)
                qx_train[1] * qx_train[2],  # (1, T_train)
                qx_train[2] * qx_train[0],  # (1, T_train)
                qx_train[0] * qx_train[1],  # (1, T_train)
            ]
        )  # (1 + Nx * 2 + 3, T_train)

        B = linalg.solve(
            (xx @ xx.T + np.eye(1 + Nx * 2 + 3)).T,  # (1 + Nx * 2 + 3, 1 + Nx * 2 + 3)
            ((np.roll(x_train, -1, axis=1) - x_train) @ xx.T).T,  # (1 + Nx * 2 + 3, Nx)
        ).T  # (Nx, 1 + Nx * 2 + 3)
        qB = linalg.solve(
            (qxx @ qxx.T + np.eye(1 + Nx * 2 + 3)).T,  # (1 + Nx * 2 + 3, 1 + Nx * 2 + 3)
            ((np.roll(qx_train, -1, axis=1) - qx_train) @ qxx.T).T,  # (1 + Nx * 2 + 3, Nx)
        ).T  # (Nx, 1 + Nx * 2 + 3)

    else:
        raise ValueError(f"Invalid sequence type: {sequence_type}")

    return B, qB


if __name__ == "__main__":
    main(
        sequence_type=2,
        out_dir=out_dir,
        seed=0,
    )
