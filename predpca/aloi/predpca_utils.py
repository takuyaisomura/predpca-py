import numpy as np


def preproc_data(
    data: np.ndarray,  # (Ns, T_train + T_test)
    T_train: int,
    T_test: int,
    Kf_list: list[int],
    WithNoise: bool,
):
    """Create random object sequences for training and test"""
    Ns = data.shape[0]
    Kf = len(Kf_list)

    seq_len = 72
    n_seq = (T_train + T_test) // seq_len
    n_seq_train = T_train // seq_len

    # separate data into sequences and shuffle
    seq_idxs = np.random.permutation(n_seq)
    data_shuffled = data.reshape(Ns, n_seq, seq_len)[:, seq_idxs, :]  # (Ns, n_seq, seq_len)

    # target for test (noise free)
    s_target_test = np.zeros((Kf, Ns, T_test))
    for k in range(Kf):
        s_target_test[k] = np.roll(data_shuffled[:, n_seq_train:, :], -Kf_list[k], axis=-1).reshape(Ns, -1)

    # add noise to data
    # target for training and inputs for training/test may contain noise
    if WithNoise:
        sigma_noise = 2.3  # same amplitude as original input covariance
        print(f"sigma_noise = {sigma_noise}")
        data_var = data_shuffled.var(axis=(1, 2), ddof=1).mean()  # TODO: check
        print(f"averaged input variance (original)   = {data_var}")
        data_shuffled = data_shuffled + np.random.randn(*data_shuffled.shape) * sigma_noise
        data_var2 = data_shuffled.var(axis=(1, 2), ddof=1).mean()
        print(f"averaged input variance (with noise) = {data_var2}")

    # target for training
    s_target_train = np.zeros((Kf, Ns, T_train))
    for k in range(Kf):
        s_target_train[k] = np.roll(data_shuffled[:, :n_seq_train, :], -Kf_list[k], axis=-1).reshape(Ns, -1)

    # inputs
    s_train = data_shuffled[:, :n_seq_train, :]  # (Ns, n_seq_train, seq_len)
    s_test = data_shuffled[:, n_seq_train:, :]  # (Ns, n_seq_test, seq_len)

    return s_train, s_test, s_target_train, s_target_test


def predict_encoding(
    W_pca_post_opt,  # (Nu, Ns)
    se_sub,  # (Kf, Ns, T)
    se_test,  # (Kf, Ns, T_test)
):
    Nu = W_pca_post_opt.shape[0]
    Kf, _, T = se_sub.shape
    T_test = se_test.shape[-1]

    u_sub_ = np.zeros((Nu, T))  # mean predictive encoders (training)
    u_test_ = np.zeros((Nu, T_test))  # mean predictive encoders (test)
    for k in range(Kf):
        u_sub_ += W_pca_post_opt @ se_sub[k] / Kf
        u_test_ += W_pca_post_opt @ se_test[k] / Kf

    u_sub_mean = u_sub_.mean(axis=1, keepdims=True)
    u_sub_ = u_sub_ - u_sub_mean
    u_test_ = u_test_ - u_sub_mean

    # deviation of predictive encoders
    du_sub = np.empty((Kf, Nu, T))
    du_test = np.empty((Kf, Nu, T_test))
    for k in range(Kf):
        du_sub[k] = W_pca_post_opt @ se_sub[k] - u_sub_  # (Nu, T_sub)
        du_test[k] = W_pca_post_opt @ se_test[k] - u_test_  # (Nu, T_test)

    return u_sub_, u_test_, du_sub, du_test


def prediction_error(
    s_target,  # (Kf, Ns, T2)
    s_estim,  # (Kf, Ns, T2)
    C,  # (Ns, Ns)
):
    Kf, Ns, T = s_target.shape
    err = np.zeros((Kf, Ns))

    for k in range(Kf):
        StSt = s_target[k] @ s_target[k].T / T  # (Ns, Ns)
        StSe = s_target[k] @ s_estim[k].T / T  # (Ns, Ns)
        SeSe = s_estim[k] @ s_estim[k].T / T  # (Ns, Ns)

        lambda1 = np.diag(C.T @ (2 * StSe - SeSe) @ C)  # (Ns,)
        lambda2 = np.zeros(Ns)
        lambda2[0] = lambda1[0]
        for i in range(1, Ns):
            lambda2[i] = lambda2[i - 1] + lambda1[i]

        err[k, :] = np.trace(StSt) - lambda2

    return err
