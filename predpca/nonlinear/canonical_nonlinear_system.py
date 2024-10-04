import numpy as np
from scipy import linalg
from tqdm import tqdm


def _generate_x_and_psi(
    Nx: int,
    Npsi: int,
    T: int,
    z: np.ndarray,
    R: np.ndarray,
    B: np.ndarray,
):
    x = np.zeros((Nx, T))
    psi = np.zeros((Npsi, T))

    x[:, 0] = np.random.randn(Nx)
    psi[:, 0] = np.tanh(R @ np.append(x[:, 0], 1))  # (Npsi, )

    for t in range(1, T):
        x[:, t] = B @ psi[:, t - 1] + z[:, t]
        psi[:, t] = np.tanh(R @ np.append(x[:, t], 1))

    return x, psi


def _compute_whitening_matrix(x: np.ndarray):
    return linalg.inv(linalg.sqrtm(np.cov(x)))

    # eigvals, eigvecs = linalg.eigh(np.cov(x))
    # return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def canonical_nonlinear_system(
    Nx: int,
    Npsi: int,
    T_train: int,
    T_test: int,
    sigma_z: float,
):
    x_test = np.zeros((Nx, T_test))
    psi_test = np.zeros((Npsi, T_test))
    z_train = np.random.randn(Nx, T_train) * sigma_z
    z_test = np.random.randn(Nx, T_test) * sigma_z
    R = np.random.randn(Npsi, Nx + 1) / np.sqrt(Nx + 1)
    B = np.random.randn(Nx, Npsi) / np.sqrt(Npsi)

    for _ in tqdm(range(100), desc="Generating data"):
        x_train, psi_train = _generate_x_and_psi(Nx, Npsi, T_train, z_train, R, B)
        # update B
        whitening_matrix = _compute_whitening_matrix(x_train[:, T_train // 10 - 1 :])
        mean_psi = psi_train.mean(axis=1, keepdims=True)
        B = B * 0.9 + whitening_matrix @ B * 0.1 - B @ mean_psi @ mean_psi.T * 0.1

    x_train, psi_train = _generate_x_and_psi(Nx, Npsi, T_train, z_train, R, B)
    x_test, psi_test = _generate_x_and_psi(Nx, Npsi, T_test, z_test, R, B)

    # whiten x
    meanx0 = np.mean(x_train, axis=1, keepdims=True)
    whitening_matrix = _compute_whitening_matrix(x_train)
    x_train = whitening_matrix @ (x_train - meanx0)
    x_test = whitening_matrix @ (x_test - meanx0)

    return (
        x_train,  # (Nx, T)
        x_test,  # (Nx, T2)
        psi_train,  # (Npsi, T)
        psi_test,  # (Npsi, T2)
    )


if __name__ == "__main__":
    canonical_nonlinear_system(10, 100, 100000, 100000, 0.001)
