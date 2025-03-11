import logging
import multiprocessing as mp
import time
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from tqdm import tqdm

from predpca.utils import DeviceManager
from predpca.utils import pcacov_torch as pcacov

torch.set_grad_enabled(False)

start_time = time.time()

nx1 = 160  # video image width
ny1 = 80  # video image height
Ndata1 = nx1 * ny1
num_vid = 10  # number of videos used for training (about 10 h each)

batch_size = 100000

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)

device_manager = DeviceManager()
num_processes = device_manager.num_gpu if device_manager.num_gpu > 0 else 2

logging.basicConfig(
    format="%(asctime)s - %(processName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main(
    overwrite: bool = False,
    seed: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)

    file_ids_to_process = [
        fileid for fileid in range(num_vid) if overwrite or not (out_dir / f"pca_lv1_{fileid:02d}.npz").exists()
    ]
    # with mp.Pool(processes=num_processes) as pool:
    #     pool.map(process_video, file_ids_to_process)
    for fileid in file_ids_to_process:
        process_video(fileid)

    logging.info(f"Merge results (time = {(time.time() - start_time) / 60:.1f} min)")
    device = device_manager.get_device()
    T = 0
    mean1 = torch.zeros((Ndata1, 1), dtype=torch.float32, device=device)
    Cov1 = torch.zeros((Ndata1, Ndata1), dtype=torch.float32, device=device)
    for fileid in tqdm(range(num_vid)):
        npz = np.load(out_dir / f"pca_lv1_{fileid:02d}.npz")
        T += npz["num_frame"]
        mean1 += torch.from_numpy(npz["mean1"]).to(device)
        Cov1 += torch.from_numpy(npz["Cov1"]).to(device)

    mean1 /= T * 12
    Cov1 /= T * 12
    Cov1 -= mean1 @ mean1.T

    PCA_C1, PCA_L1, _ = pcacov(Cov1)

    logging.info(f"Save data (time = {(time.time() - start_time) / 60:.1f} min)")
    np.savez_compressed(
        out_dir / "pca_lv1_dst.npz",
        mean1=mean1.cpu().numpy(),  # (Ndata1, 1)
        PCA_C1=PCA_C1.cpu().numpy(),  # (Ndata1, Ndata1)
        PCA_L1=PCA_L1.cpu().numpy(),  # (Ndata1, 1)
        T=T,
    )

    device_manager.release_device(device)


def process_video(
    fileid: int,
) -> None:
    mp.current_process().name = f"Video-{fileid:02d}"
    device = device_manager.get_device()

    logging.info(f"Processing with device {device}")
    path = data_dir / f"train{fileid:02d}.mp4"

    # get the number of frames
    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num_frame = int(video_info["nb_frames"])

    data_buffer = torch.empty((ny1 * 2, nx1 * 2, batch_size, 3), dtype=torch.uint8, device=device)
    mean1 = torch.zeros((Ndata1, 1), dtype=torch.float32, device=device)
    Cov1 = torch.zeros((Ndata1, Ndata1), dtype=torch.float32, device=device)

    num_batches = np.ceil(num_frame / batch_size).astype(int)
    for i_batch in range(num_batches):
        logging.info(f"Batch {i_batch + 1}/{num_batches} (time = {(time.time() - start_time) / 60:.1f} min)")

        start_frame = i_batch * batch_size
        end_frame = min((i_batch + 1) * batch_size, num_frame)
        batch_mean1, batch_Cov1 = process_batch(path, start_frame, end_frame, device, data_buffer)

        mean1 += batch_mean1
        Cov1 += batch_Cov1

    logging.info(f"Save data (time = {(time.time() - start_time) / 60:.1f} min)")
    np.savez_compressed(
        out_dir / f"pca_lv1_{fileid:02d}.npz",
        mean1=mean1.cpu().numpy(),
        Cov1=Cov1.cpu().numpy(),
        num_frame=num_frame,
    )

    device_manager.release_device(device)


def process_batch(
    path: Path,
    start_frame: int,
    end_frame: int,
    device: torch.device,
    data_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # read frames
    read_length = end_frame - start_frame
    data = data_buffer[:, :, :read_length, :]

    ffmpeg_command = (
        ffmpeg.input(path, ss=start_frame / 30)  # Assuming 30 fps, adjust if different
        .filter("select", f"gte(n,{0})")
        .filter("select", f"lte(n,{read_length - 1})")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=read_length)
    )
    if device.type == "cuda":
        ffmpeg_command = ffmpeg_command.global_args("-hwaccel", "cuda")

    out, _ = ffmpeg_command.run(capture_stdout=True, quiet=True)
    data.copy_(torch.frombuffer(out, dtype=torch.uint8).view(read_length, ny1 * 2, nx1 * 2, 3).permute(1, 2, 0, 3))

    data[:ny1, :, :, :] = torch.flip(data[:ny1, :, :, :], dims=[0])
    data[:, :nx1, :, :] = torch.flip(data[:, :nx1, :, :], dims=[1])

    # Compute mean and cov
    mean1 = torch.zeros((Ndata1, 1), dtype=torch.float32, device=device)
    Cov1 = torch.zeros((Ndata1, Ndata1), dtype=torch.float32, device=device)

    for i in range(2):
        for j in range(2):
            s = (
                data[ny1 * i : ny1 * (i + 1), nx1 * j : nx1 * (j + 1)].permute(3, 2, 1, 0).reshape(-1, Ndata1).T.float()
                / 255.0
            )
            mean1 += s.sum(dim=1, keepdim=True)
            Cov1 += torch.matmul(s, s.T)

    return mean1, Cov1


if __name__ == "__main__":
    main()
