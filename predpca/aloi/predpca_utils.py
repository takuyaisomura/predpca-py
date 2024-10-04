import numpy as np


def preproc_data(
    data: np.ndarray,  # (Ns, T_train + T_test)
    T_train: int,
    T_test: int,
    Kf: int,
    Kp: int,
    Kp2: int,
    WithNoise: bool,
):
    Ns = data.shape[0]
    # Create random object sequences for training and test
    order = np.random.permutation((T_train + T_test) // 72)

    Kf_list = [6, 12, 18, 24, 30]
    Kp_list = list(range(0, 37, 2))
    Kp2_list = list(range(37))

    ts_target = np.zeros((Kf, T_train + T_test), dtype=int)
    ts_input1 = np.zeros((Kp, T_train + T_test), dtype=int)
    ts_input2 = np.zeros((Kp2, T_train + T_test), dtype=int)

    for i in range((T_train + T_test) // 72):
        j = order[i] * 72
        t = i * 72
        for k in range(Kf):
            ts_target[k, t : t + 72] = np.roll(np.arange(j, j + 72), -Kf_list[k])
        for k in range(Kp):
            ts_input1[k, t : t + 72] = np.roll(np.arange(j, j + 72), Kp_list[k])
        for k in range(Kp2):
            ts_input2[k, t : t + 72] = np.roll(np.arange(j, j + 72), Kp2_list[k])

    # target for test (target for test is noise free)
    s_target_test = np.array([data[:, ts_target[i, T_train:]] for i in range(Kf)])  # (Kf, Ns, T_test)

    # target for training and inputs for training and test may contain noise
    if WithNoise:
        sigma_noise = 2.3  # same amplitude as original input covariance
        print(f"sigma_noise = {sigma_noise}")
        data_var = data.var(axis=1, ddof=1).mean()
        print(f"averaged input variance (original)   = {data_var}")
        data = data + np.random.randn(Ns, T_train + T_test) * sigma_noise
        data_var2 = data.var(axis=1, ddof=1).mean()
        print(f"averaged input variance (with noise) = {data_var2}")

    # target for training
    s_train_target = np.array([data[:, ts_target[i, :T_train]] for i in range(Kf)])  # (Kf, Ns, T_train)
    # inputs
    s_train = data[:, ts_input1[0, :T_train]]  # training input data (Ns, T_train)

    return s_train, s_train_target, s_target_test, ts_input1, ts_input2


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
