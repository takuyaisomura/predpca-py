import time
from pathlib import Path

import numpy as np
from PIL import Image

from predpca.utils import pcacov

# Initialization
start_time = time.time()
Timg = 72000  # number of images
Nimgx = 192  # original image width
Nimgy = 144  # original image height
nx1 = 72  # half of image width
ny1 = 72  # half of image height
Ndata1 = nx1 * ny1 * 3  # input dimensionality of level 1 PCA
Npca1 = 1000  # output dimensionality of level 1 PCA
Ndata2 = Npca1 * 4  # input dimensionality of level 2 PCA
Npca2 = 1000  # output dimensionality of level 2 PCA

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output"


def main(
    data_dir: Path,
    out_dir: Path,
):
    print("----------------------------------------")
    print("Preprocess ALOI dataset")
    print("----------------------------------------\n")

    # Level 1 PCA
    print(f"level 1 PCA (time = {(time.time() - start_time) / 60:.1f} min)")
    img = np.zeros((Nimgy, Nimgx, 3, 72), dtype=np.uint8)
    data = np.zeros((Nimgy, Nimgy, 3, Timg), dtype=np.uint8)

    print(f"read image files (time = {(time.time() - start_time) / 60:.1f} min)")

    N_obj = Timg // 72
    for n_obj in range(1, N_obj + 1):
        if n_obj % 100 == 0:
            print(f"n_obj = {n_obj} / {N_obj} (time = {(time.time() - start_time) / 60:.1f} min)")

        for n_ang in range(72):
            img[:, :, :, n_ang] = np.array(Image.open(data_dir / f"png4/{n_obj}/{n_obj}_r{n_ang * 5}.png"))

        data[:, :, :, 72 * (n_obj - 1) : 72 * n_obj] = img[:, 24:168, :, :]

    Cov1 = np.empty((2, 2, Ndata1, Ndata1), dtype=np.float32)
    mean1 = np.empty((2, 2, Ndata1), dtype=np.float32)
    PCA_C1 = np.empty((2, 2, Ndata1, Ndata1), dtype=np.float32)
    PCA_L1 = np.empty((2, 2, Ndata1), dtype=np.float32)

    for i in range(2):
        for j in range(2):
            print(f"compute covariance (time = {(time.time() - start_time) / 60:.1f} min)")
            s = (
                data[ny1 * i : ny1 * (i + 1), nx1 * j : nx1 * (j + 1), :, :]  # (ny1, nx1, 3, Timg)
                .reshape(Ndata1, Timg, order="F")
                .astype(np.float32)
                / 255
            )  # (Ndata1, Timg)
            mean1[i, j] = np.mean(s, axis=1)  # (Ndata1,)
            Cov1[i, j] = np.cov(s)  # (Ndata1, Ndata1)
            print(f"eigenvalue decomposition (time = {(time.time() - start_time) / 60:.1f} min)")
            PCA_C1[i, j], PCA_L1[i, j], _ = pcacov(Cov1[i, j])

    print("----------------------------------------\n")

    # Level 2 PCA
    print(f"level 2 PCA (time = {(time.time() - start_time) / 60:.1f} min)")
    s = np.zeros((Ndata2, Timg), dtype=np.float32)

    print(f"compute covariance (time = {(time.time() - start_time) / 60:.1f} min)")

    for i in range(2):
        for j in range(2):
            W1 = PCA_C1[i, j, :, :Npca1].T  # (Npca1, Ndata1)
            s[Npca1 * (2 * i + j) : Npca1 * (2 * i + j + 1), :] = W1 @ (
                data[ny1 * i : ny1 * (i + 1), nx1 * j : nx1 * (j + 1), :, :]
                .reshape(Ndata1, Timg, order="F")
                .astype(np.float32)
                / 255
                - mean1[i, j, :, np.newaxis]
            )  # (Npca1, Timg)

    mean2 = np.mean(s, axis=1)  # (Ndata2,)
    Cov2 = np.cov(s)  # (Ndata2, Ndata2)

    print(f"eigenvalue decomposition (time = {(time.time() - start_time) / 60:.1f} min)")
    PCA_C2, PCA_L2, _ = pcacov(Cov2)  # PCA_C2: (Ndata2, Ndata2), PCA_L2: (Ndata2,)

    print("----------------------------------------\n")

    # Save data
    print(f"save data (time = {(time.time() - start_time) / 60:.1f} min)")
    W2 = PCA_C2[:, :Npca2].T  # (Npca2, Ndata2)
    data = W2 @ (s - mean2[:, np.newaxis])  # (Npca2, Timg)
    np.savez_compressed(
        out_dir / "aloi_preprocessed.npz",
        mean1=mean1,  # (2, 2, Ndata1)
        PCA_C1=PCA_C1,  # (2, 2, Ndata1, Ndata1)
        PCA_L1=PCA_L1,  # (2, 2, Ndata1)
        mean2=mean2,  # (Ndata2,)
        PCA_C2=PCA_C2,  # (Ndata2, Ndata2)
        PCA_L2=PCA_L2,  # (Ndata2,)
        data=data,  # (Npca2, Timg)
    )

    print(f"complete preprocessing (time = {(time.time() - start_time) / 60:.1f} min)")
    print("----------------------------------------\n")


if __name__ == "__main__":
    main(data_dir, out_dir)
