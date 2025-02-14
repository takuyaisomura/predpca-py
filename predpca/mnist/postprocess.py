import numpy as np
from scipy import linalg, stats
from tqdm import tqdm

ica_rep = 2000  # number of iteration for ICA
ica_eta = 0.01  # learning rate for ICA


def postprocessing(
    u_train: np.ndarray,  # (Nu, T_train)
    u_test: np.ndarray,  # (Nu, T_test)
    label_test: np.ndarray,  # (1, T_test)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Nv = np.unique(label_test).size

    u_std = np.diag(np.std(u_train, axis=1, ddof=1))  # (Nu, Nu)
    u_train = linalg.inv(u_std) @ u_train  # (Nu, T_train)
    u_test = linalg.inv(u_std) @ u_test  # (Nu, T_test)

    # ICA
    # ui_train: (Nu, T_train)
    # ui_test: (Nu, T_test)
    # Wica: (Nu, Nu)
    ui_train, ui_test, Wica = ica(u_train, u_test, ica_rep, ica_eta)

    # adjust the sign of components to make the skewness positive
    skewness = stats.skew(ui_test, axis=1)
    ui_train *= np.sign(skewness)[:, np.newaxis]
    ui_test *= np.sign(skewness)[:, np.newaxis]

    # normalize variance
    u_std = np.diag(np.std(ui_train, axis=1, ddof=1))  # (Nu, Nu)
    coeff = np.sqrt(0.1) * linalg.inv(u_std)  # (Nu, Nu)
    ui_train = coeff @ ui_train  # (Nu, T_train)
    ui_test = coeff @ ui_test  # (Nu, T_test)

    # winner-takes-all
    v_test = (ui_test == ui_test.max(axis=0)).astype(int)  # (Nv, T_test)

    # categorization error
    conf_mat = np.zeros((Nv, Nv))  # confusion matrix (true vs pred)
    for i in range(Nv):
        conf_mat[i, :] = np.sum(v_test[:, label_test[0] == i], axis=1)

    categorization_error = np.mean(1 - conf_mat.max(axis=0) / (conf_mat.sum(axis=0) + 0.001))
    print(f"categorization error = {categorization_error}")

    return ui_train, ui_test, Wica, v_test, conf_mat


def ica(
    u_train: np.ndarray,  # (Nu, T_train)
    u_test: np.ndarray,  # (Nu, T_test)
    ica_rep: int,
    ica_eta: float,
):
    T = u_train.shape[1]
    Nu = u_train.shape[0]
    rndn = np.random.randn(Nu, Nu)
    Wica, _, _ = linalg.svd(rndn)  # (Nu, Nu)

    sample_size = T // 10
    for _ in tqdm(range(ica_rep), desc="ICA"):
        rnd = np.random.randint(T, size=sample_size)
        ui_train = Wica @ u_train[:, rnd]  # ICA encoders (Nu, sample_size)
        g = np.sqrt(2) * np.tanh(100 * ui_train)  # nonlinear activation function (Nu, sample_size)
        Wica += ica_eta * (Wica - (g @ ui_train.T / sample_size) @ Wica)  # Amari's ICA rule

    ui_train = Wica @ u_train  # (Nu, T_train)
    ui_test = Wica @ u_test  # (Nu, T_test)

    return ui_train, ui_test, Wica
