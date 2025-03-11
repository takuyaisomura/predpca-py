import logging
import time
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from matplotlib import pyplot as plt

from predpca.utils import pcacov_torch as pcacov
from predpca.utils import sym_positive

torch.set_grad_enabled(False)

start_time = time.time()

nx1 = 160  # video image width
ny1 = 80  # video image height
Ndata1 = nx1 * ny1
Npca1 = 2000  # dimensionality of input fed to PredPCA
Nppca = [
    2000,
    1600,
    1600,
    1600,
    1600,
    1600,
]
Nu = 100

batch_size = 50000
num_max_batches = 8

# predict observation at t+Kf based on observations between t-Kp+1 and t
# 30 step = 1 s
Kp = 8  # order of past observations
Kf = 15  # interval
decay = 0.1

Mixture = True  # using mixture model

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(
    preproc_out_dir: Path = out_dir,
    train_out_dir: Path = out_dir,
):
    # compute optimal matrices for predictions
    mle_data = np.load(train_out_dir / "mle_lv1_dst.npz")
    pca_data = np.load(preproc_out_dir / "pca_lv1_dst.npz")
    PCA_C1 = torch.from_numpy(pca_data["PCA_C1"]).to(device)  # (Ndata1, Ndata1)
    PCA_L1 = torch.from_numpy(pca_data["PCA_L1"]).to(device)  # (Ndata1,)
    mean1 = torch.from_numpy(pca_data["mean1"]).to(device)  # (Ndata1, 1)
    Wpca = PCA_C1[:, :Npca1].T  # (Npca1, Ndata1)

    logging.info(f"Compute Q (time = {(time.time() - start_time) / 60:.1f} min)")

    STS = {
        part: torch.from_numpy(mle_data[f"STS_{part}"]).double().to(device) for part in ["t", "b"]
    }  # STS[part][i]: (Npca1, Npca1 * Kp)
    S_S = {
        part: torch.from_numpy(mle_data[f"S_S_{part}"]).double().to(device) for part in ["t", "b"]
    }  # S_S[part][i]: (Npca1 * Kp, Npca1 * Kp)
    Tpart = {part: torch.from_numpy(mle_data[f"Tpart_{part}"]).to(device) for part in ["t", "b"]}
    eye = torch.eye(Npca1 * Kp, device=device)
    if Mixture:
        for part in ["t", "b"]:
            for i in range(6):
                STS[part][i] /= Tpart[part][i]
                S_S[part][i] = sym_positive(S_S[part][i] / Tpart[part][i]) + eye * 1e-6

        Q = {
            part: [torch.linalg.solve(S_S[part][i], STS[part][i], left=False) for i in range(6)]  # (Npca1, Npca1 * Kp)
            for part in ["t", "b"]
        }
        SESE = {
            part: sum(Q[part][i] @ S_S[part][i] @ Q[part][i].T for i in range(6)).float() / 6  # (Npca1, Npca1)
            for part in ["t", "b"]
        }
    else:
        for part in ["t", "b"]:
            STS[part] = STS[part].sum(dim=0) / Tpart[part].sum()
            S_S[part] = sym_positive(S_S[part].sum(dim=0) / Tpart[part].sum()) + eye * 1e-6

        Q = {part: torch.linalg.solve(S_S[part], STS[part], left=False) for part in ["t", "b"]}
        SESE = {part: (Q[part] @ S_S[part] @ Q[part].T).float() for part in ["t", "b"]}

    # post-hoc PCA
    logging.info("Post-hoc PCA")
    PPCA_C1 = {}
    PPCA_L1 = {}
    for part in ["t", "b"]:
        PPCA_C1[part], PPCA_L1[part], _ = pcacov(sym_positive(SESE[part]))

    np.savez_compressed(
        out_dir / "predpca_lv1_dst.npz",
        **{f"PPCA_C1{part}": PPCA_C1[part].cpu().numpy() for part in ["t", "b"]},
        **{f"PPCA_L1{part}": PPCA_L1[part].cpu().numpy() for part in ["t", "b"]},
    )

    logging.info("Compute Lambda")
    if Mixture:
        Lambda = {
            part: torch.stack(
                [pcacov(sym_positive(Q[part][i] @ S_S[part][i] @ Q[part][i].T))[1] for i in range(6)]  # (Npca1,)
            )  # (6, Npca1)
            for part in ["t", "b"]
        }
    else:
        Lambda = {part: pcacov(sym_positive(Q[part] @ S_S[part] @ Q[part].T))[1] for part in ["t", "b"]}

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, part, title in zip(axes, ["t", "b"], ["top", "bottom"]):
        ax.plot(range(1, Npca1 + 1), torch.log10(Lambda[part].T).cpu().numpy())
        ax.plot(range(1, Npca1 + 1), torch.log10(PCA_L1[:Npca1]).cpu().numpy(), "k--")
        ax.set_title(title)
        ax.set_xlabel("Component")
        ax.set_ylabel("Log10(Eigenvalue)")
    fig.savefig(out_dir / "predpca_bdd100k_test.png")

    del PCA_C1, STS, S_S, SESE, PPCA_L1, Lambda, eye
    torch.cuda.empty_cache()

    # dimensionality reduction
    logging.info("Dimensionality reduction")
    if Mixture:
        Q = {
            part: [
                PPCA_C1[part][:, : Nppca[i]].double()
                @ PPCA_C1[part][:, : Nppca[i]].double().T
                @ Q[part][i]  # (Npca1, Npca1 * Kp)
                for i in range(6)
            ]
            for part in ["t", "b"]
        }
    else:
        Q = {
            part: PPCA_C1[part][:, : Nppca[0]].double()
            @ PPCA_C1[part][:, : Nppca[0]].double().T
            @ Q[part]  # (Npca1, Npca1 * Kp)
            for part in ["t", "b"]
        }

    Wppca = {part: PPCA_C1[part][:, :Nu].T for part in ["t", "b"]}  # (Nu, Npca1)

    del PPCA_C1
    torch.cuda.empty_cache()

    # predict 0.5 s future image of test data
    fileid = 0
    path = data_dir / f"test{fileid:02d}.mp4"
    logging.info(f"Processing video {fileid} with device {device}")

    # Use ffmpeg to get video information
    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    num_frames = int(video_info["nb_frames"])
    num_batches = min(num_max_batches, np.ceil(num_frames / batch_size).astype(int))

    err = np.zeros((2, batch_size, num_batches))
    amp = np.zeros((2, batch_size, num_batches))
    data_buffer = torch.empty((ny1 * 2, nx1 * 2, batch_size, 3), dtype=torch.uint8, device=device)

    for k in range(num_batches):
        logging.info(f"{k + 1}/{num_batches} (time = {(time.time() - start_time) / 60:.1f} min)")

        start_frame = k * batch_size
        end_frame = min((k + 1) * batch_size, num_frames)
        err0, amp0, amp1, u = process_batch(
            path,
            start_frame,
            end_frame,
            Wpca,
            mean1,
            PCA_L1,
            Q,
            Wppca,
            data_buffer,
        )

        err1 = err0
        logging.info(f"error = {err0.mean() / amp0.mean()}")
        logging.info(f"error = {err1.mean() / amp1.mean()}")
        err[0, :, k] = np.pad(err0, (0, batch_size - len(err0)), constant_values=np.nan)
        err[1, :, k] = np.pad(err1, (0, batch_size - len(err1)), constant_values=np.nan)
        amp[0, :, k] = np.pad(amp0, (0, batch_size - len(amp0)), constant_values=np.nan)
        amp[1, :, k] = np.pad(amp1, (0, batch_size - len(amp1)), constant_values=np.nan)

        np.savez_compressed(out_dir / f"predpca_lv1_u_{k + 1}.npz", u=u)

    np.savetxt(
        out_dir / "predpca_test_prediction_error.csv",
        # np.vstack([np.nanmean(err, axis=1), np.nanmean(amp, axis=1)]),  # (4, num_batches)
        np.vstack(
            [
                np.nanmean(err[0, :, :], axis=0),
                np.nanmean(amp[0, :, :], axis=0),
                np.nanmean(err[1, :, :], axis=0),
                np.nanmean(amp[1, :, :], axis=0),
            ]
        ),  # (4, num_batches)
        delimiter=",",
    )


def process_batch(
    path: Path,
    start_frame: int,
    end_frame: int,
    Wpca: torch.Tensor,
    mean1: torch.Tensor,
    PCA_L1: torch.Tensor,
    Q: dict[str, list[torch.Tensor]] | dict[str, torch.Tensor],
    Wppca: dict[str, torch.Tensor],
    data_buffer: torch.Tensor,
):
    read_length = end_frame - start_frame
    data = data_buffer[:, :, :read_length, :]

    ffmpeg_command = (
        ffmpeg.input(path, ss=start_frame / 30)  # Assuming 30 fps
        .filter("select", f"gte(n,{0})")
        .filter("select", f"lte(n,{read_length - 1})")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=read_length)
    )
    if device.type == "cuda":
        ffmpeg_command = ffmpeg_command.global_args("-hwaccel", "cuda", "-hwaccel_device", str(device.index))

    out, _ = ffmpeg_command.run(capture_stdout=True, quiet=True)
    data.copy_(torch.frombuffer(out, dtype=torch.uint8).view(read_length, ny1 * 2, nx1 * 2, 3).permute(1, 2, 0, 3))
    del out

    data[:ny1, :, :, :] = torch.flip(data[:ny1, :, :, :], dims=[0])
    data[:, :nx1, :, :] = torch.flip(data[:, :nx1, :, :], dims=[1])

    logging.info("Compute predicted inputs")

    err0 = torch.zeros(read_length, device=device)
    amp0 = torch.zeros(read_length, device=device)
    amp1 = torch.zeros(read_length, device=device)
    u_list = torch.zeros((2, 2, Nu, read_length * 3))

    # compute top left, top right, bottom left, bottom right areas
    for i in range(2):
        part = "t" if i == 0 else "b"
        for j in range(2):
            logging.info(f"{i=}, {j=}")

            # sensory input
            s = Wpca @ (
                data[ny1 * i : ny1 * (i + 1), nx1 * j : nx1 * (j + 1), :, :]
                .permute(3, 2, 1, 0)
                .reshape(-1, Ndata1)
                .T.float()
                / 255
                - mean1
            )  # (Npca1, read_length * 3)
            # target
            st = torch.roll(s, -Kf, dims=1)
            # basis
            phi = torch.diag(1 / torch.sqrt(PCA_L1[:Npca1])) @ s  # (Npca1, read_length * 3)
            phi1 = torch.cat([torch.roll(phi, k, dims=1) for k in range(Kp)], dim=0)  # (Npca1 * Kp, read_length * 3)

            if Mixture:
                # predicted input
                ses = torch.zeros((6, Npca1, read_length * 3), device=device)
                # prediction error under each model
                G = torch.zeros((6, read_length * 3), device=device)

                for h in range(6):
                    ses[h] = Q[part][h] @ phi1.double()  # (Npca1, read_length * 3)
                    G[h, :] = torch.mean((torch.roll(ses[h], Kf, dims=1) - s) ** 2, dim=0)

                flag = G.T.reshape(3, read_length, 6).mean(dim=0).T  # (6, read_length)
                flag = (flag == torch.min(flag, dim=0)[0]).double()
                for t in range(1, read_length):
                    flag[:, t] = flag[:, t - 1] + decay * (-flag[:, t - 1] + flag[:, t])
                flag = flag.tile(1, 3)  # (6, read_length * 3)
                se = sum(ses[h] * flag[h : h + 1, :] for h in range(6))
            else:
                se = Q[part] @ phi1.double()  # (Npca1, read_length * 3)

            del phi, phi1, ses, G, flag
            torch.cuda.empty_cache()

            u = Wppca[part].double() @ se  # (Nu, read_length * 3)
            e1 = ((st - se) ** 2).sum(dim=0).reshape(3, read_length).sum(dim=0).mean()
            e2 = ((st - s) ** 2).sum(dim=0).reshape(3, read_length).sum(dim=0).mean()
            logging.info(f"err_part = {e1 / e2}")

            err0 += ((st - se) ** 2).sum(dim=0).reshape(3, read_length).sum(dim=0)
            amp0 += (st**2).sum(dim=0).reshape(3, read_length).sum(dim=0)
            amp1 += ((st - s) ** 2).sum(dim=0).reshape(3, read_length).sum(dim=0)

            u_list[i, j] = u

    logging.info("Compute test prediction error")

    return (
        err0.cpu().numpy(),
        amp0.cpu().numpy(),
        amp1.cpu().numpy(),
        u_list.numpy(),
    )


if __name__ == "__main__":
    main()
