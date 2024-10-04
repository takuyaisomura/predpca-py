import numpy as np
from scipy import linalg


def calculate_estimated_parameters(
    s_train: np.ndarray,  # (Ns, T)
    u_train: np.ndarray,  # (Nu, T)
    uc_train: np.ndarray,  # (Nu, T)
    W_pca_post: np.ndarray,  # (Nu, Ns)
    Omega_inv: np.ndarray,  # (Nu, Nu)
    prior_psi: float,
):
    T = s_train.shape[1]
    Nu = u_train.shape[0]
    v_train = Omega_inv @ u_train
    vc_train = Omega_inv @ uc_train
    qA = W_pca_post.T @ linalg.inv(Omega_inv + np.eye(Nu) * 1e-4)  # observation matrix (Ns, Nu)
    qB = (np.roll(v_train, -1, axis=1) @ v_train.T) @ linalg.inv(
        v_train @ v_train.T + np.eye(Nu) * prior_psi
    )  # transition matrix (Ns, Nu)
    qSigmas = (s_train @ s_train.T) / T  # actual input covariance (Ns, Ns)
    qSigmap = linalg.inv(qB + np.eye(Nu) * 1e-4) @ (np.roll(vc_train, -1, axis=1) @ vc_train.T / T)
    qSigmap = (qSigmap + qSigmap.T) / 2  # hidden basis covariance (Ns, Ns)
    qSigmaz = qSigmap - qB @ qSigmap @ qB.T  # system noise covariance (Ns, Ns)
    qSigmao = qSigmas - qA @ qSigmap @ qA.T  # observation noise covariance (Ns, Ns)

    return qA, qB, qSigmas, qSigmap, qSigmao, qSigmaz


def calculate_true_parameters(
    s,  # (Ns, T)
    x,  # (Nx, T)
    prior_x,  # int
):
    T = s.shape[1]
    Nx = x.shape[0]

    A = s @ x.T @ linalg.inv(x @ x.T + np.eye(Nx) * prior_x)  # (Ns, Nx)
    B = np.roll(x, -1, axis=1) @ x.T @ linalg.inv(x @ x.T + np.eye(Nx) * prior_x)  # (Nx, Nx)

    Sigmas = s @ s.T / T  # (Ns, Ns)
    Sigmap = x @ x.T / T  # (Nx, Nx)
    Sigmao = Sigmas - A @ Sigmap @ A.T  # (Ns, Ns)
    Sigmaz = Sigmap - B @ Sigmap @ B.T  # (Nx, Nx)

    return A, B, Sigmas, Sigmap, Sigmao, Sigmaz


def generate_true_hidden_states(
    input,  # (784, T)
    label,  # (1, T)
):
    Nx = 10
    T = label.shape[1]

    x = np.zeros((Nx, T))
    for i in range(Nx):
        x[i, :] = label == i

    x *= np.sign(np.sum(input, axis=0, keepdims=True))

    return x
