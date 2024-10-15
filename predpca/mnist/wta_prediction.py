import numpy as np
from scipy import linalg


def wta_prediction(
    ui_train: np.ndarray,  # (Nu, T_train)
    v_test: np.ndarray,  # (Nv, T_test)
    input_train: np.ndarray,  # (N, T_train)
    input_mean: np.ndarray,  # (N, 1)
    prior_u: float,
    prior_u_: float,
    order: int,
):
    # initialization
    Nu = ui_train.shape[0]
    T_train = ui_train.shape[1]
    T_test = v_test.shape[1]

    eye = np.eye(Nu, dtype=int)

    def onehot(x):  # x: (N, T)
        return eye[:, x.argmax(axis=0)]  # (N, T)

    A = input_train @ ui_train.T @ linalg.inv(ui_train @ ui_train.T + np.eye(Nu) * prior_u)  # (N, Nu)

    ui_train = np.abs(ui_train)
    v_train = onehot(ui_train)

    ui_shifted = create_shifted_arrays(ui_train, order)
    ui_ = create_multistep_state(ui_shifted, order)

    # optimal transition matrix
    B = ui_shifted[order] @ ui_.T @ linalg.inv(ui_ @ ui_.T + np.eye(Nu**order) * prior_u_)  # (Nu, Nu)
    ui_pred = v_test.copy()

    for t in range(60, T_test):
        ui_pred[:, t] = B @ create_multistep_state_t(ui_pred, order, t)  # state transition (Nu, 1)
        ui_pred[:, t] = onehot(ui_pred[:, t])  # winner-takes-all

    v_shifted = create_shifted_arrays(v_train, order)
    v_train_ = create_multistep_state(v_shifted, order)
    log = np.log(linalg.det(np.cov((v_shifted[order] - B @ v_train_))))
    AIC = T_train * log + 2 * Nu * Nu**order

    # visualize prediction results
    input_pred = A @ ui_pred + input_mean  # (N, T_test)

    return input_pred, ui_pred, A, B, AIC


def create_shifted_arrays(array, model_order):
    return [np.roll(array, -i, axis=1) for i in range(model_order + 1)]


def create_multistep_state(ui_shifted, order):
    Nu = ui_shifted[0].shape[0]
    if order == 1:
        return ui_shifted[0]  # (Nu, T)
    elif order == 2:
        return ui_shifted[1].repeat(Nu, axis=0) * np.tile(ui_shifted[0], (Nu, 1))  # (Nu**2, T)
    elif order == 3:
        return (
            ui_shifted[2].repeat(Nu**2, axis=0)
            * np.tile(ui_shifted[1].repeat(Nu, axis=0), (Nu, 1))
            * np.tile(ui_shifted[0], (Nu**2, 1))
        )  # (Nu**3, T)
    elif order == 4:
        return (
            ui_shifted[3].repeat(Nu**3, axis=0)
            * np.tile(ui_shifted[2].repeat(Nu**2, axis=0), (Nu, 1))
            * np.tile(ui_shifted[1].repeat(Nu, axis=0), (Nu**2, 1))
            * np.tile(ui_shifted[0], (Nu**3, 1))
        )  # (Nu**4, T)


def create_multistep_state_t(ui, order, t):
    if order == 1:
        return ui[:, t - 1]  # (Nu,)
    elif order == 2:
        return np.kron(ui[:, t - 1], ui[:, t - 2])  # (Nu**2,)
    elif order == 3:
        return np.kron(
            np.kron(ui[:, t - 1], ui[:, t - 2]),
            ui[:, t - 3],
        )  # (Nu**3,)
    elif order == 4:
        return np.kron(
            np.kron(ui[:, t - 1], ui[:, t - 2]),
            np.kron(ui[:, t - 3], ui[:, t - 4]),
        )  # (Nu**4,)
