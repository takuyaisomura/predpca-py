import logging
import time
from pathlib import Path

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from scipy import linalg, stats
from sklearn.decomposition import PCA
from tqdm import tqdm

start_time = time.time()

nx1 = 160  # video image width
ny1 = 80  # video image height
Ndata1 = nx1 * ny1
Npca1 = 2000  # dimensionality of input fed to PredPCA
Ns = Npca1
Nv = 100
Nu = 100
test_num_batches = 8  # number of batches in main_test.py
vid_frames = 10000  # number of video frames to output

batch_size = 100000
num_max_batches = 4

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)


def main(
    data_dir: Path,
    out_dir: Path,
    preproc_out_dir: Path | None = None,
    test_out_dir: Path | None = None,
    seed: int = 2,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    preproc_out_dir = preproc_out_dir or out_dir
    test_out_dir = test_out_dir or out_dir

    np.random.seed(1000000 + seed)

    # Load data
    logging.info("Loading data")
    u = load_u(test_out_dir)

    # PCA
    logging.info("PCA")
    pca = PCA(n_components=Nv)
    up = pca.fit_transform((u - np.roll(u, 1, axis=1)).T).T  # (Nv, T)
    mean_up = up.mean(axis=1, keepdims=True)  # (Nv, 1)
    std_up = up.std(axis=1, ddof=1)  # (Nv,)
    up = np.diag(1 / std_up) @ (up - mean_up)  # (Nv, T)

    np.random.seed(1000000 + seed)  # reset seed because PCA uses random numbers

    # ICA
    logging.info("ICA")
    Wica, _ = ica(up, device)
    Wica = Wica.cpu().numpy()

    # Output independent components
    logging.info("Output independent components")
    uica = np.diag(std_up) @ linalg.inv(Wica) * 40 + mean_up
    out = plot_ica(preproc_out_dir, test_out_dir, uica, pca.components_.T, out_dir)

    # Compute features
    logging.info("Compute features")
    fileid = 0
    path = data_dir / f"test{fileid:02d}.mp4"
    lateral_motion = compute_features(path)

    # Analyze PC1 of dynamical features
    logging.info("Analyze PC1 of dynamical features")
    analyze_pcs(up, lateral_motion, out_dir)

    # Show movie
    logging.info("Show movie")
    data = read_batch(path, 0, vid_frames)
    show_video(data, up, out_dir)

    logging.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")

    return (
        u,
        Wica,
        uica,
        out,
        up,
        lateral_motion,
    )


def load_u(test_out_dir: Path):
    u_lists = [
        np.load(test_out_dir / f"predpca_lv1_u_{i + 1}.npz")["u"] for i in range(test_num_batches)
    ]  # (test_num_batches, 2, 2, Nu, test_batch_size * 3)
    u_lists = [
        np.vstack([ls[i][j] for i in range(2) for j in range(2)]) for ls in u_lists
    ]  # (test_num_batches, Nu * 4, test_batch_size * 3)
    u_lists = [
        ls.reshape(Nu * 4, -1, 3, order="F").transpose(0, 2, 1).reshape(Nu * 4 * 3, -1, order="F") for ls in u_lists
    ]  # (test_num_batches, Nu * 12, test_batch_size)
    u = np.hstack(u_lists)  # (Nu * 12, test_batch_size * test_num_batches)
    return u


def ica(up, device):
    ica_rep = 20000
    T = up.shape[1]
    sample_size = T // 10

    up = torch.from_numpy(up.astype(np.float32)).to(device)
    Wica = torch.eye(Nv, device=device)
    eye = torch.eye(Nv, device=device)

    for t in tqdm(range(1, ica_rep + 1), desc="ICA"):
        if t < 4000:
            eta = 0.04
        elif t < 8000:
            eta = 0.02
        elif t < 12000:
            eta = 0.01
        else:
            eta = 0.005

        t_list = np.random.randint(0, T, sample_size)  # (sample_size,)
        v = Wica @ up[:, t_list]  # (Nv, sample_size)
        g = torch.tanh(100 * v)  # (Nv, sample_size)
        Wica = Wica + eta * (eye - g @ v.T / sample_size) @ Wica  # (Nv, Nv)

    v = Wica @ up  # (Nv, T)
    skew_sign = np.sign(stats.skew(v.cpu().numpy(), axis=1).astype(np.float32))  # (Nv,)
    Wica = torch.diag(torch.from_numpy(skew_sign).to(device)) @ torch.diag(1 / v.std(dim=1)) @ Wica  # (Nv, Nv)
    v = Wica @ up  # (Nv, T)

    return Wica, v


def plot_ica(
    preproc_out_dir: Path,
    test_out_dir: Path,
    uica: np.ndarray,
    C: np.ndarray,
    out_dir: Path,
):
    pca_lv1_dst = np.load(preproc_out_dir / "pca_lv1_dst.npz")
    PCA_C1 = pca_lv1_dst["PCA_C1"]
    predpca_lv1_dst = np.load(test_out_dir / "predpca_lv1_dst.npz")
    PPCA_C1t = predpca_lv1_dst["PPCA_C1t"]
    PPCA_C1b = predpca_lv1_dst["PPCA_C1b"]

    out = np.zeros((ny1 * 2, nx1 * 2, 3, Nv), dtype=np.float32)

    PCA_PPCA_C1t = PCA_C1[:, :Ns] @ PPCA_C1t[:, :Nu]
    PCA_PPCA_C1b = PCA_C1[:, :Ns] @ PPCA_C1b[:, :Nu]
    shape = (ny1, nx1, Nv)
    out[:ny1, :nx1, 0, :] = (PCA_PPCA_C1t @ C[0 * Nu : 1 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[:ny1, nx1:, 0, :] = (PCA_PPCA_C1t @ C[1 * Nu : 2 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[ny1:, :nx1, 0, :] = (PCA_PPCA_C1b @ C[2 * Nu : 3 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[ny1:, nx1:, 0, :] = (PCA_PPCA_C1b @ C[3 * Nu : 4 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[:ny1, :nx1, 1, :] = (PCA_PPCA_C1t @ C[4 * Nu : 5 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[:ny1, nx1:, 1, :] = (PCA_PPCA_C1t @ C[5 * Nu : 6 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[ny1:, :nx1, 1, :] = (PCA_PPCA_C1b @ C[6 * Nu : 7 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[ny1:, nx1:, 1, :] = (PCA_PPCA_C1b @ C[7 * Nu : 8 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[:ny1, :nx1, 2, :] = (PCA_PPCA_C1t @ C[8 * Nu : 9 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[:ny1, nx1:, 2, :] = (PCA_PPCA_C1t @ C[9 * Nu : 10 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[ny1:, :nx1, 2, :] = (PCA_PPCA_C1b @ C[10 * Nu : 11 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[ny1:, nx1:, 2, :] = (PCA_PPCA_C1b @ C[11 * Nu : 12 * Nu] @ uica + 0.5).reshape(shape, order="F")
    out[:ny1, :, :, :] = np.flip(out[:ny1, :, :, :], axis=0)
    out[:, :nx1, :, :] = np.flip(out[:, :nx1, :, :], axis=1)
    out = np.clip(out, 0, 1)

    chunk_size = 10
    num_chunks = Nv // chunk_size
    interval = 10

    skew = stats.skew(out.reshape(ny1 * 2 * nx1 * 2 * 3, Nv, order="F"), axis=0)
    var_x = out.transpose(0, 2, 1, 3).reshape(ny1 * 2 * 3, nx1 * 2, Nv, order="F").var(axis=0, ddof=1)  # (nx1 * 2, Nv)
    var_y = out.transpose(1, 2, 0, 3).reshape(nx1 * 2 * 3, ny1 * 2, Nv, order="F").var(axis=0, ddof=1)  # (ny1 * 2, Nv)
    sorted_idx = np.argsort((np.arange(1, nx1 * 2 + 1) @ var_x) / var_x.sum(axis=0))

    for i in range(num_chunks):
        chunk = slice(chunk_size * i, chunk_size * (i + 1))
        idx2 = np.argsort(
            (np.arange(1, ny1 * 2 + 1) @ var_y[:, sorted_idx[chunk]]) / var_y[:, sorted_idx[chunk]].sum(axis=0)
        )
        sorted_idx[chunk] = sorted_idx[chunk_size * i + idx2]

    img = (
        np.ones(
            (
                ny1 * 2 * chunk_size + (chunk_size - 1) * interval,
                nx1 * 2 * chunk_size + (chunk_size - 1) * interval,
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(chunk_size):
        for j in range(num_chunks):
            top = (ny1 * 2 + interval) * i
            left = (nx1 * 2 + interval) * j
            if skew[sorted_idx[chunk_size * j + i]] > 0:
                out_ = out[:, :, :, sorted_idx[chunk_size * j + i]]
            else:
                out_ = 1 - out[:, :, :, sorted_idx[chunk_size * j + i]]
            img[top : top + ny1 * 2, left : left + nx1 * 2] = np.uint8(out_ * 255)

    dpi = 100
    plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(out_dir / "predpca_ica_dyn100.png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()

    return out


def compute_features(path):
    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num_frames = int(video_info["nb_frames"])

    num_batches = min(num_max_batches, np.ceil(num_frames / batch_size).astype(int))
    processed_frames = min(num_frames, num_batches * batch_size)

    lateral_motion = np.zeros(processed_frames)

    for k in range(num_batches):
        logging.info(f"Batch {k + 1}/{num_batches} (time = {(time.time() - start_time) / 60:.1f} min)")
        start = k * batch_size
        end = min((k + 1) * batch_size, processed_frames)
        data = read_batch(path, start, end)

        data_ = data.mean(axis=(0, 2))
        lateral_motion[start:end] = (np.roll(data_[1:, :], -1, axis=1) - data_[:-1, :]).mean(axis=0) / 255

    return lateral_motion


def read_batch(path, start_frame, end_frame):
    read_length = end_frame - start_frame

    ffmpeg_command = (
        ffmpeg.input(path, ss=start_frame / 30)
        .filter("select", f"gte(n,{0})")
        .filter("select", f"lte(n,{read_length - 1})")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=read_length)
    )

    out, _ = ffmpeg_command.run(capture_stdout=True, quiet=True)
    data = np.frombuffer(out, dtype=np.uint8).copy().reshape(read_length, ny1 * 2, nx1 * 2, 3).transpose(1, 2, 3, 0)
    # (ny1 * 2, nx1 * 2, 3, read_length)

    return data


def analyze_pcs(
    up: np.ndarray,
    lateral_motion: np.ndarray,
    out_dir: Path,
):
    num_bins = 100
    prctile_pc1 = np.zeros((3, num_bins))
    count = np.zeros(num_bins)

    for i in range(num_bins):
        idx = (lateral_motion >= i * 0.0004 - 0.02) & (lateral_motion < (i + 1) * 0.0004 - 0.02)
        count[i] = np.sum(idx)
        if count[i] > 0:
            prctile_pc1[:, i] = np.percentile(up[0, idx], [25, 50, 75])
        else:
            prctile_pc1[:, i] = np.nan

    x = np.arange(1, num_bins + 1) * 0.0004 - 0.02
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, prctile_pc1.T)
    ax.plot(x, count / count.sum())
    ax.set_xlim(-0.01, 0.01)
    ax.set_xticks([-0.01, 0, 0.01])
    ax.set_xlabel("Lateral motion")
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("PC1")
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    plt.tight_layout()
    fig.savefig(out_dir / "predpca_dyn_analysis.png")

    np.savetxt(
        out_dir / "predpca_dyn_pc1_lateral_motion.csv",
        np.vstack([x, prctile_pc1, count]),
        delimiter=",",
    )


def show_video(
    data: np.ndarray,
    up: np.ndarray,
    out_dir: Path,
):
    num_frames = data.shape[3]
    time_v = np.argsort(np.abs(up[:, :num_frames]), axis=1)[::-1]

    fps = 10
    dpi = 100
    output_file = out_dir / "predpca_bdd_dyn.mp4"

    writer = FFMpegWriter(fps=fps)

    fig = plt.figure(figsize=(20, 11))
    gs = plt.GridSpec(2, 4, figure=fig, left=0, right=1, top=0.95, bottom=0, wspace=0, hspace=0)
    axs = [fig.add_subplot(gs[i, j]) for j in range(4) for i in range(2)]

    with writer.saving(fig, output_file, dpi):
        for t in tqdm(range(num_frames), desc="Generating video"):
            for i, ax in enumerate(axs):
                ax.clear()
                ax.imshow(data[:, :, :, time_v[i][t]])
                ax.axis("off")
                ax.set_aspect("auto")
            fig.suptitle(f"Frame {t}", y=0.98, fontsize=20)
            writer.grab_frame()

            for i, ax in enumerate(axs):
                ax.clear()
                if time_v[i][t] + 1 >= data.shape[-1]:
                    continue
                ax.imshow(data[:, :, :, time_v[i][t] + 1])
                ax.axis("off")
                ax.set_aspect("auto")
            fig.suptitle(f"Frame {t}", y=0.98, fontsize=20)
            writer.grab_frame()

    plt.close(fig)
    logging.info(f"Video saved to {output_file}")


if __name__ == "__main__":
    main(
        data_dir=data_dir,
        out_dir=out_dir,
    )
