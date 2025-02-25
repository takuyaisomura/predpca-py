import time
from pathlib import Path

import numpy as np

from predpca.aloi.predpca_utils import predict_encoding, prepare_train_test_sequences
from predpca.aloi.visualize import plot_hidden_state, plot_true_and_pred_video
from predpca.models import PredPCA

start_time = time.time()

T_train = 57600  # number of training data
T_test = 14400  # number of test data

Ns = 300  # dimensionality of inputs
Nu = 128  # dimensionality of encoders
Nv = 20  # number of hidden states to visualize

Kp_list = range(0, 37, 2)  # past timepoints to be used for basis functions
Kf_list = [6, 12, 18, 24, 30]  # future timepoints to be used for prediction targets
Kf_viz = 2  # Kf_viz-th timepoint in Kf_list will be visualized
WithNoise = False  # presence of noise

# Priors (small constants) to prevent inverse matrix from being singular
prior_s = 100.0
prior_s_ = 100.0

out_dir = Path(__file__).parent / "output" / Path(__file__).stem


def main(
    out_dir: Path,
    preproc_out_dir: Path = Path(__file__).parent / "output",
    seed: int = 0,
):
    print("----------------------------------------")
    print("PredPCA of 3D rotating object images")
    print("----------------------------------------\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)

    print("loading data")
    npz = np.load(preproc_out_dir / "aloi_preprocessed.npz")
    data = npz["data"][:Ns, :].astype(float)  # (Ns, Timg)

    print("preprocessing")
    s_train, s_test, s_target_train, s_target_test = prepare_train_test_sequences(
        data, T_train, T_test, Kf_list, WithNoise
    )

    # PredPCA
    print(f"create basis functions: {(time.time() - start_time) / 60:.1f} min")
    predpca = PredPCA(kp_list=Kp_list, prior_s_=prior_s_)
    se_train = predpca.fit_transform(s_train, s_target_train)  # (Kf, Ns, T_train)
    se_test = predpca.transform(s_test)  # (Kf, Ns, T_test)

    W_pca_post = predpca.predict_input_pca(se_train)[0].T
    u_test_viz = W_pca_post @ se_test[Kf_viz]  # predictive encoders (test) (Ns, T_test)
    u_train, _, _, _ = predict_encoding(W_pca_post, se_train, se_test)

    # visualizations
    PCA_C1 = npz["PCA_C1"]  # (2, 2, Ndata1, Ndata1)
    PCA_C2 = npz["PCA_C2"]  # (Ndata2, Ndata2)
    mean1 = npz["mean1"]  # (2, 2, Ndata1)

    # true and predicted images (for Fig 3a and Suppl Movie)
    print(f"true and predicted images (time = {(time.time() - start_time) / 60:.1f} min)")
    print("create supplementary movie")
    plot_true_and_pred_video(W_pca_post, u_test_viz, s_target_test, PCA_C1, PCA_C2, mean1, out_dir)
    print("----------------------------------------\n")

    # hidden state analysis
    # ICA of mean encoders (for Fig 3b and Suppl Fig 4)
    print(f"ICA of mean encoders (time = {(time.time() - start_time) / 60:.1f} min)")
    plot_hidden_state(W_pca_post, u_train, PCA_C1, PCA_C2, mean1, Nv, WithNoise, out_dir)

    print("----------------------------------------\n")

    return u_test_viz, u_train


if __name__ == "__main__":
    main(
        out_dir=out_dir,
        seed=0,
    )
