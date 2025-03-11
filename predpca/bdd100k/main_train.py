import logging
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import ffmpeg
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

from predpca.utils import DeviceManager

start_time = time.time()

torch.set_grad_enabled(False)
# disable_tqdm = False
disable_tqdm = True  # for parallel processing

nx1 = 160  # video image width
ny1 = 80  # video image height
Ndata1 = nx1 * ny1
Npca1 = 2000  # dimensionality of input fed to PredPCA
num_vid = 20  # number of videos used for training (about 10 h each)

# Predict observation at t+Kf based on observations between t-Kp+1 and t
# 30 step = 1 s
Kp = 8  # order of past observations
Kf = 15  # interval

batch_size = 50000

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)

device_manager = DeviceManager()
num_processes = device_manager.num_gpu if device_manager.num_gpu > 0 else 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    preproc_out_dir: Path = out_dir,
    overwrite: bool = False,
):
    logging.info(f"maximum likelihood estimation (time = {(time.time() - start_time) / 60:.1f} min)")

    # Load PCA results
    pca_data = np.load(preproc_out_dir / "pca_lv1_dst.npz")
    PCA_C1 = pca_data["PCA_C1"]
    PCA_L1 = pca_data["PCA_L1"]
    mean1 = pca_data["mean1"]
    Wpca = PCA_C1[:, :Npca1].T

    file_ids_to_process = [
        fileid for fileid in range(num_vid) if overwrite or not (out_dir / f"mle_lv1_{fileid:02d}.npz").exists()
    ]
    logging.info(f"Processing videos: {file_ids_to_process}")
    with mp.Pool(processes=num_processes) as pool:
        process_video_partial = partial(process_video, Wpca_np=Wpca, PCA_L1_np=PCA_L1, mean1_np=mean1)
        pool.map(process_video_partial, file_ids_to_process)
    # for fileid in file_ids_to_process:
    #     process_video(fileid, Wpca, PCA_L1, mean1)

    logging.info(f"Merge results (time = {(time.time() - start_time) / 60:.1f} min)")
    STS, S_S, Tpart = merge_results(num_vid)

    logging.info(f"Save data (time = {(time.time() - start_time) / 60:.1f} min)")
    np.savez_compressed(
        out_dir / "mle_lv1_dst.npz",
        **{f"STS_{part}": STS[part] for part in ["t", "b"]},
        **{f"S_S_{part}": S_S[part] for part in ["t", "b"]},
        **{f"Tpart_{part}": Tpart[part] for part in ["t", "b"]},
    )


def merge_results(num_vid: int):
    STS = {part: np.zeros((6, Npca1, Npca1 * Kp), dtype=np.float32) for part in ["t", "b"]}
    S_S = {part: np.zeros((6, Npca1 * Kp, Npca1 * Kp), dtype=np.float32) for part in ["t", "b"]}
    Tpart = {part: np.zeros(6, dtype=np.float32) for part in ["t", "b"]}

    for fileid in tqdm(range(num_vid), desc="Merging results"):
        path = out_dir / f"mle_lv1_{fileid:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")

        data = np.load(path)
        for part in ["t", "b"]:
            STS[part] += data[f"STS_{part}"]
            S_S[part] += data[f"S_S_{part}"]
            Tpart[part] += data[f"Tpart_{part}"]

    return STS, S_S, Tpart


def process_video(
    fileid: int,
    Wpca_np: np.ndarray,
    PCA_L1_np: np.ndarray,
    mean1_np: np.ndarray,
):
    mp.current_process().name = f"Video-{fileid:02d}"
    device = device_manager.get_device()

    Wpca = torch.from_numpy(Wpca_np).to(device)
    PCA_L1 = torch.from_numpy(PCA_L1_np).to(device)
    mean1 = torch.from_numpy(mean1_np).to(device)

    logging.info(f"Processing video with device {device}")
    path = data_dir / f"train{fileid:02d}.mp4"

    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num_frames = int(video_info["nb_frames"])
    num_batches = np.ceil(num_frames / batch_size).astype(int)

    # Maximum likelihood estimation
    STS = {part: torch.zeros((6, Npca1, Npca1 * Kp)) for part in ["t", "b"]}
    S_S = {part: torch.zeros((6, Npca1 * Kp, Npca1 * Kp)) for part in ["t", "b"]}
    Tpart = {part: torch.zeros(6, dtype=torch.int32) for part in ["t", "b"]}

    data_buffer = torch.empty((ny1 * 2, nx1 * 2, batch_size, 3), dtype=torch.uint8, device=device)

    for k in range(num_batches):
        logging.info(f"Batch {k + 1}/{num_batches} (time = {(time.time() - start_time) / 60:.1f} min)")

        start_frame = k * batch_size
        end_frame = min((k + 1) * batch_size, num_frames)

        batch_STS, batch_S_S, batch_Tpart = process_batch(
            path,
            start_frame,
            end_frame,
            Wpca,
            PCA_L1,
            mean1,
            data_buffer,
            device,
        )
        for part in ["t", "b"]:
            STS[part] += batch_STS[part]
            S_S[part] += batch_S_S[part]
            Tpart[part] += batch_Tpart[part]

    logging.info(f"Saving data (time = {(time.time() - start_time) / 60:.1f} min)")
    np.savez_compressed(
        out_dir / f"mle_lv1_{fileid:02d}.npz",
        **{f"STS_{part}": STS[part].cpu().numpy() for part in ["t", "b"]},
        **{f"S_S_{part}": S_S[part].cpu().numpy() for part in ["t", "b"]},
        **{f"Tpart_{part}": Tpart[part].cpu().numpy() for part in ["t", "b"]},
    )

    device_manager.release_device(device)


def process_batch(
    path: Path,
    start_frame: int,
    end_frame: int,
    Wpca: torch.Tensor,
    PCA_L1: torch.Tensor,
    mean1: torch.Tensor,
    data_buffer: torch.Tensor,
    device: torch.device,
):
    read_length = end_frame - start_frame
    data = data_buffer[:, :, :read_length, :]

    # Read video frames using ffmpeg
    ffmpeg_command = (
        ffmpeg.input(path, ss=start_frame / 30)  # Assuming 30 fps, adjust if different
        .filter("select", f"gte(n,{0})")
        .filter("select", f"lte(n,{read_length - 1})")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=read_length)
    )
    if device.type == "cuda":
        ffmpeg_command = ffmpeg_command.global_args("-hwaccel", "cuda", "-hwaccel_device", str(device.index))

    out, _ = ffmpeg_command.run(capture_stdout=True, quiet=True)
    data.copy_(torch.frombuffer(out, dtype=torch.uint8).view(read_length, ny1 * 2, nx1 * 2, 3).permute(1, 2, 0, 3))
    del out

    # pre-compute resized data list
    logging.info("Resizing and flipping")
    Kf2 = Kf // 3
    wx = torch.arange(6, device=device) * nx1 * 2 * Kf2 // 50 // 2
    wy = torch.arange(6, device=device) * ny1 * 2 * Kf2 // 50 // 2
    resized_data_list = torch.zeros(6, ny1, nx1, read_length, 3, dtype=torch.uint8)
    resize = transforms.Resize(size=(ny1, nx1), interpolation=transforms.InterpolationMode.BICUBIC)
    for m in tqdm(range(6), desc="Resizing and flipping", disable=disable_tqdm):
        resized_data = resize(
            data[
                wy[m] : ny1 * 2 - wy[m],
                wx[m] : nx1 * 2 - wx[m],
            ].permute(2, 3, 0, 1)
        ).permute(2, 3, 0, 1)
        resized_data.clamp_(0, 255)
        resized_data_list[m] = resized_data
    del resized_data
    torch.cuda.empty_cache()

    # pre-compute target and bases
    logging.info("Flipping")
    data[:ny1, :, :, :] = torch.flip(data[:ny1, :, :, :], dims=[0])
    data[:, :nx1, :, :] = torch.flip(data[:, :nx1, :, :], dims=[1])

    logging.info("Computing target and bases")
    st_list = torch.zeros(2, 2, Npca1, read_length * 3)
    phi1_list = torch.zeros(2, 2, Npca1 * Kp, read_length * 3)
    for i in range(2):
        for j in range(2):
            logging.info(f"Computing target and bases {i=}, {j=}")
            # Sensory input
            s = Wpca @ (
                data[ny1 * i : ny1 * (i + 1), nx1 * j : nx1 * (j + 1), :, :]
                .permute(3, 2, 1, 0)
                .reshape(-1, Ndata1)
                .T.float()
                / 255
                - mean1
            )  # (Npca1, Td * 3)
            # Target
            st = torch.roll(s, -Kf, dims=1)  # (Npca1, Td * 3)
            st_list[i, j] = st
            # Bases
            phi = torch.diag(1 / PCA_L1[:Npca1].sqrt()) @ s  # (Npca1, Td * 3)
            phi1 = torch.cat([torch.roll(phi, k, dims=1) for k in range(Kp)], dim=0)  # (Npca1 * Kp, Td * 3)
            phi1_list[i, j] = phi1
    del data, s, phi
    del st, phi1
    torch.cuda.empty_cache()

    # result placeholder in cpu
    STS = {part: torch.zeros((6, Npca1, Npca1 * Kp)) for part in ["t", "b"]}
    S_S = {part: torch.zeros((6, Npca1 * Kp, Npca1 * Kp)) for part in ["t", "b"]}
    Tpart = {part: torch.zeros(6, dtype=torch.int32) for part in ["t", "b"]}

    G = torch.zeros((6, read_length * 3))  # cpu
    for i in range(2):
        for j in range(2):
            logging.info(f"Training {i=}, {j=}")

            for m in tqdm(range(6), desc="Categorization based on speed", disable=disable_tqdm):
                # Categorization based on speed
                data_ = resized_data_list[m].to(device)
                ss = (
                    data_[ny1 // 2 * i : ny1 // 2 * (i + 1), nx1 // 2 * j : nx1 // 2 * (j + 1), :, :]
                    .permute(3, 2, 1, 0)
                    .reshape(-1, Ndata1 // 4)
                    .T.float()
                    / 255
                )  # (Ndata1 // 4, Td * 3)

                if m == 0:
                    sst = torch.roll(ss, -Kf, dims=1)

                G[m, :] = torch.mean((ss - sst) ** 2, dim=0)

                del ss
                torch.cuda.empty_cache()

            del sst
            torch.cuda.empty_cache()

            flag = G == torch.min(G, dim=0, keepdim=True)[0]
            part = "t" if i == 0 else "b"
            Tpart[part] += torch.sum(flag, dim=1)

            # Synaptic update
            for m in tqdm(range(6), desc="Synaptic update", disable=disable_tqdm):
                st = st_list[i, j, :, flag[m, :]].to(device)
                phi1 = phi1_list[i, j, :, flag[m, :]].to(device)
                STS[part][m] += (st @ phi1.T).cpu()  # (Npca1, Npca1 * Kp)
                S_S[part][m] += (phi1 @ phi1.T).cpu()  # (Npca1 * Kp, Npca1 * Kp)

            del flag, st, phi1
            torch.cuda.empty_cache()

    return STS, S_S, Tpart


if __name__ == "__main__":
    main()
