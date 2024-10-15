import numpy as np
from scipy import linalg

from predpca.utils.pcacov import pcacov


def create_basis_functions(
    s_train: np.ndarray,  # (Ns, T_train)
    s_test: np.ndarray,  # (Ns, T_test)
    Kp_list: list[int] | range,  # past timepoints to be used for basis functions
    gain: np.ndarray | None = None,  # (Nu, Ns)
):
    # multi_seq with n_seq = 1
    return create_basis_functions_multi_seq(s_train[:, np.newaxis, :], s_test[:, np.newaxis, :], Kp_list, gain)


def create_basis_functions_multi_seq(
    s_train: np.ndarray,  # (Ns, n_seq_train, seq_len)
    s_test: np.ndarray,  # (Ns, n_seq_test, seq_len)
    Kp_list: list[int] | range,  # past timepoints to be used for basis functions
    gain: np.ndarray | None = None,  # (Nu, Ns)
) -> tuple[np.ndarray, np.ndarray]:
    if gain is not None:
        s_train = np.einsum("ij, jkl -> ikl", gain, s_train)  # (Nu, n_seq_train, seq_len)
        s_test = np.einsum("ij, jkl -> ikl", gain, s_test)  # (Nu, n_seq_test, seq_len)

    enc_dim, n_seq_train, seq_len_train = s_train.shape
    _, n_seq_test, seq_len_test = s_test.shape
    Kp = len(Kp_list)

    s_train_ = np.zeros((enc_dim * Kp, n_seq_train * seq_len_train))
    s_test_ = np.zeros((enc_dim * Kp, n_seq_test * seq_len_test))
    for i, k in enumerate(Kp_list):
        s_train_[enc_dim * i : enc_dim * (i + 1), :] = np.roll(s_train, k, axis=-1).reshape(enc_dim, -1)
        s_test_[enc_dim * i : enc_dim * (i + 1), :] = np.roll(s_test, k, axis=-1).reshape(enc_dim, -1)

    return s_train_, s_test_


def compute_Q(
    s_train_: np.ndarray,  # (N*Kp, T)
    s_train_target: np.ndarray,  # (Kf, Ns, T) | (Ns, T)
    S_S_inv: np.ndarray | None = None,  # (N*Kp, N*Kp)
    prior_s_: float | int | None = None,
) -> np.ndarray:
    if s_train_target.ndim == 2:  # accept 2d input if Kf == 1
        s_train_target = s_train_target[np.newaxis]  # (1, Ns, T)

    Kf, Ns, _ = s_train_target.shape
    Nphi = s_train_.shape[0]

    if S_S_inv is None:
        assert isinstance(prior_s_, (float, int))
        S_S_inv = linalg.inv(s_train_ @ s_train_.T + np.eye(Nphi) * prior_s_)
    else:
        assert isinstance(S_S_inv, np.ndarray)

    Q = np.empty((Kf, Ns, Nphi))
    for k in range(Kf):
        Q[k] = s_train_target[k] @ s_train_.T @ S_S_inv  # (Ns, N*Kp)

    if Kf == 1:  # squeeze if Kf == 1
        Q = Q[0]

    return Q  # (Kf, Ns, Ns*Kp) | (Ns, Ns*Kp)


def predict_input(
    Q,  # (Kf, Ns, Ns*Kp) | (Ns, Ns*Kp)
    s_,  # (Ns*Kp, T)
):
    if Q.ndim == 2:  # accept 2d input if Kf == 1
        Q = Q[np.newaxis]  # (1, Ns, Ns*Kp)

    Kf, Ns, _ = Q.shape
    T = s_.shape[1]

    se = np.empty((Kf, Ns, T))
    for k in range(Kf):
        se[k] = Q[k] @ s_  # (Ns, T)

    if Kf == 1:  # squeeze if Kf == 1
        se = se[0]

    return se  # (Kf, Ns, T) | (Ns, T)


def predict_input_pca(
    se_train,  # (Kf, Ns, T)
):
    if se_train.ndim == 2:  # accept 2d input if Kf == 1
        se_train = se_train[np.newaxis]  # (1, Ns, T)

    Kf, Ns, T = se_train.shape

    # predict input covariance
    qSigmase = np.empty((Kf, Ns, Ns))
    for k in range(Kf):
        qSigmase[k] = se_train[k] @ se_train[k].T / T  # (Ns, Ns)

    qSigmase_mean = qSigmase.mean(axis=0)

    C, L, _ = pcacov(qSigmase_mean)  # C: (Ns, Ns), L: (Ns,)

    return C, L, qSigmase_mean
