from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy import linalg, stats
from skimage.transform import rescale
from tqdm import tqdm


def plot_true_and_pred_video(
    W_pca_post_opt: np.ndarray,  # (Nu, Ns)
    u_test: np.ndarray,  # (Nu, T_test)
    s_target_test: np.ndarray,  # (Kf, Ns, T_test)
    PCA_C1: np.ndarray,  # (2, 2, Ndata1, Ndata1)
    PCA_C2: np.ndarray,  # (Ndata2, Ndata2)
    mean1: np.ndarray,  # (2, 2, Ndata1)
    out_dir: Path,
):
    Ns = W_pca_post_opt.shape[1]

    err_obj = np.zeros(200)
    var_obj = np.zeros(200)
    for t in range(200):
        err_obj[t] = (
            np.square(s_target_test[2, :, t * 72 : (t + 1) * 72] - W_pca_post_opt.T @ u_test[:, t * 72 : (t + 1) * 72])
            .sum(axis=0)
            .mean()
        )
        var_obj[t] = s_target_test[2, :, t * 72 : (t + 1) * 72].var(axis=1).sum()

    idx = np.argsort(err_obj / var_obj)  # (200,)

    numx = 20
    numy = 10

    frames = []
    for rot in tqdm(range(72), desc="Video frames"):
        s_list = np.zeros((Ns, 400))
        s_list[:, ::2] = s_target_test[2][:, rot + idx * 72]
        s_list[:, 1::2] = W_pca_post_opt.T @ u_test[:, rot + idx * 72]
        img = state_to_image(s_list[:, :200], PCA_C2, PCA_C1, mean1, numx, numy)
        img = rescale(img, 0.5, anti_aliasing=True, channel_axis=2)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        frames.append(img)

    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(str(out_dir / "predpca_movie.mp4"))


def plot_hidden_state(
    W_pca_post_opt: np.ndarray,  # (Nu, Ns)
    u_sub: np.ndarray,  # (Nu, T_train)
    PCA_C1: np.ndarray,  # (2, 2, Ndata1, Ndata1)
    PCA_C2: np.ndarray,  # (Ndata2, Ndata2)
    mean1: np.ndarray,  # (2, 2, Ndata1)
    Nv: int,
    WithNoise: bool,
    out_dir: Path,
):
    T_train = u_sub.shape[1]

    Wica = np.diag(1 / u_sub[:Nv, :].std(axis=1, ddof=1))  # (Nv, Nv)

    ica_rep = 8000
    sample_size = T_train // 10
    for t in tqdm(range(1, ica_rep + 1), desc="ICA"):
        if t < 2000:
            eta = 0.02
        elif t < 4000:
            eta = 0.01
        else:
            eta = 0.005

        t_list = np.random.randint(0, T_train, sample_size)  # T_sub
        v_sub = Wica @ u_sub[:Nv, t_list]  # (Nv, T // 10)
        g_sub = np.tanh(100 * v_sub)
        Wica = Wica + eta * (np.eye(Nv) - g_sub @ v_sub.T / sample_size) @ Wica  # (Nv, Nv)

    v_sub = Wica @ u_sub[:Nv, :]  # (Nv, T)
    idx = np.argsort(stats.kurtosis(v_sub, axis=1))[::-1]  # (Nv,)
    Omega = np.zeros((Nv, Nv))
    Omega[np.arange(Nv), idx] = 1
    Wica = Omega @ np.diag(np.sign(stats.skew(v_sub, axis=1))) @ Wica  # (Nv, Nv)

    if WithNoise:
        v_sub_ref = np.load(out_dir / "v_sub_ref.npy")
    else:
        v_sub_ref = v_sub
        np.save(out_dir / "v_sub_ref.npy", v_sub_ref)

    Omega = np.corrcoef(v_sub_ref, v_sub)[:Nv, Nv:]  # (Nv, Nv)

    for i in range(Nv):
        j = np.argmax(np.abs(Omega[i, :]))
        temp = np.sign(Omega[i, j])
        Omega[i, :] = 0
        Omega[:, j] = 0
        Omega[i, j] = temp

    Wica = Omega @ Wica  # (Nv, Nv)

    # images corresponding to independent components
    ic_to_obs = W_pca_post_opt[:Nv, :].T @ linalg.inv(Wica) * 20
    img = state_to_image(
        ic_to_obs,
        PCA_C2,
        PCA_C1,
        mean1,
        int(np.ceil(Nv / 3)),
        3,
    )
    img = np.uint8(np.clip(img, 0, 1) * 255)
    plt.imsave(out_dir / "predpca_ica.png", img)
    print("----------------------------------------\n")


def plot_test_images(
    W_pca_post_opt,  # (Nu, Ns)
    u_test,  # (Nu, T_test)
    PCA_C1,  # (2, 2, Ndata1, Ndata1)
    PCA_C2,  # (Ndata2, Ndata2)
    mean1,  # (2, 2, Ndata1)
    out_dir,
):
    for i in range(10):
        img = state_to_image(
            W_pca_post_opt.T @ u_test[:, np.arange(4) * 18 + i * 72],
            PCA_C2,
            PCA_C1,
            mean1,
            4,
            1,
        )
        img = np.uint8(np.clip(img, 0, 1) * 255)
        plt.imsave(out_dir / f"test_img_{i + 1}.png", img)


def state_to_image(
    sp,  # (Ns, T)
    PCA_C2,  # (Ndata2, Ndata2)
    PCA_C1,  # (2, 2, Ndata1, Ndata1)
    mean1,  # (2, 2, Ndata1)
    numx: int,
    numy: int,
):
    Ns = sp.shape[0]
    T = sp.shape[1]
    nx1 = 72  # half of image width
    ny1 = 72  # half of image height
    Npca1 = 1000
    data = np.zeros((ny1 * 2, nx1 * 2, 3, T))
    # Ndata1 = nx1 * ny1 * 3
    # Ndata2 = Npca1 * 4

    for row in range(2):
        for col in range(2):
            i = 2 * row + col
            data2 = PCA_C2[Npca1 * i : Npca1 * (i + 1), :Ns] @ sp  # (1000, T)
            data[ny1 * row : ny1 * (row + 1), nx1 * col : nx1 * (col + 1), :, :] = np.reshape(
                PCA_C1[row, col][:, :Npca1] @ data2 + mean1[row, col][:, np.newaxis],  # (Ndata1, T)
                (ny1, nx1, 3, T),
                order="F",
            )

    img = np.ones(
        (
            (ny1 * 2) * numy + 16 * (numy - 1),
            (nx1 * 2) * numx + 16 * (numx - 1),
            3,
        )
    )

    for t in range(T):
        row, col = divmod(t, numx)
        top = 160 * row
        left = 160 * col
        img[top : top + ny1 * 2, left : left + nx1 * 2, :] = data[:, :, :, t]

    return img
