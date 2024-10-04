import numpy as np
from scipy import linalg


def _generate_lorenz(
    T: int,
    y_init: np.ndarray,
):
    num_vars = 3
    eps = 0.01
    p = 10
    r = 28
    b = 8 / 3

    y = np.zeros((num_vars, T))
    y[:, 0] = y_init

    for t in range(1, T):
        y[0, t] = y[0, t - 1] + eps * (-p * (y[0, t - 1] - y[1, t - 1]))
        y[1, t] = y[1, t - 1] + eps * (-y[0, t - 1] * y[2, t - 1] + r * y[0, t - 1] - y[1, t - 1])
        y[2, t] = y[2, t - 1] + eps * (y[0, t - 1] * y[1, t - 1] - b * y[2, t - 1])

    return y


def lorenz_attractor(
    Nx: int,
    T_train: int,
    T_test: int,
):
    num_vars = 3
    Nl = Nx // num_vars
    x_train = np.zeros((Nx, T_train))
    x_test = np.zeros((Nx, T_test))

    x_train[:, 0] = np.random.rand(Nx) * 10
    for i in range(Nl):
        chunk_slice = slice(num_vars * i, num_vars * (i + 1))
        x_train[chunk_slice, :] = _generate_lorenz(T_train, x_train[chunk_slice, 0])

    x_test[:, 0] = x_train[:, -1]
    for i in range(Nl):
        chunk_slice = slice(num_vars * i, num_vars * (i + 1))
        x_test[chunk_slice, :] = _generate_lorenz(T_test, x_test[chunk_slice, 0])

    meanx0 = x_train.mean(axis=1, keepdims=True)  # (Nx, 1)
    whitening_matrix = linalg.inv(linalg.sqrtm(np.cov(x_train)))  # (Nx, Nx)
    x_train = whitening_matrix @ (x_train - meanx0)  # (Nx, T)
    x_test = whitening_matrix @ (x_test - meanx0)  # (Nx, T2)

    return x_train, x_test
