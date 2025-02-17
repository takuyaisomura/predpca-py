from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from predpca.mnist.create_digit_sequence import create_digit_sequence
from predpca.mnist.postprocess import postprocessing
from predpca.mnist.predpca_utils import generate_true_hidden_states
from predpca.mnist.visualize import digit_image, visualize_encodings
from predpca.mnist.wta_prediction import wta_prediction
from predpca.models import PredPCA

train_randomness = True
test_randomness = False
train_signflip = True
test_signflip = False
T_train = 100000  # training sample size
T_test = T_train + 1000  # test sample size

# magnitude of regularization term
prior_s_ = 100.0
prior_u = 1.0
prior_u_ = 1.0
prior_x = 1.0

Ns = 40  # input dimensionality
Nu = 10  # encoding dimensionality
Kp = 40  # order of past observations used for prediction
Nv = 10  # encoding dimensionality

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output" / Path(__file__).stem


def main(
    sequence_type: int,  # 1=ascending, 2=Fibonacci
    data_dir: Path,
    out_dir: Path,
    max_model_order: int = 4,
    seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(1000000 + seed)

    seq_label = "ascending" if sequence_type == 1 else "fibonacci"

    # create input sequences
    print("read files")
    input_train, input_test, _, label_train, label_test, _ = create_digit_sequence(
        data_dir,
        sequence_type,
        T_train,
        T_test,
        T_test,
        train_randomness,
        test_randomness,
        train_signflip,
        test_signflip,
    )
    # input: (784, T), label: (1, T)
    input_mean = input_train.mean(axis=1, keepdims=True)  # (784, 1)
    input_train = input_train - input_mean
    input_test = input_test - input_mean

    print("compress data using PCA as preprocessing")
    pca = PCA(n_components=Ns)
    s_train = pca.fit_transform(input_train.T).T  # (Ns, T_train)
    s_test = pca.transform(input_test.T).T  # (Ns, T_test)

    # compute true states
    x = generate_true_hidden_states(input_train, label_train)  # (Nx=10, T_train)
    x = x - x.mean(axis=1, keepdims=True)

    # PredPCA
    print("PredPCA")
    print("- compute maximum likelihood estimator")
    predpca = PredPCA(kp_list=range(1, Kp + 1), prior_s_=prior_s_)
    se_train = predpca.fit_transform(s_train, s_train)  # (Ns, T_train)
    se_test = predpca.transform(s_test)  # (Ns, T_test)

    print("- post-hoc PCA using eigenvalue decomposition")
    pca = PCA(n_components=Nu)
    u_train = pca.fit_transform(se_train.T).T  # (Nu, T_train)
    u_test = pca.transform(se_test.T).T  # (Nu, T_test)

    # ICA
    ui_train, ui_test, _, v_test, conf_mat = postprocessing(u_train, u_test, label_test)
    # ui_train: (Nu, T_train)
    # ui_test: (Nu, T_test)
    # v_test: (Nv, T_test)
    # conf_mat: (Nv, Nv)

    fig = visualize_encodings(ui_test, label_test, range(T_test // 10))
    fig.savefig(out_dir / f"output_encoders_{seq_label}_{seed}.png")

    data_file = np.vstack((np.arange(10), 1 - conf_mat.max(axis=0) / (conf_mat.sum(axis=0) + 0.001)))
    np.savetxt(out_dir / f"output_err_{seq_label}_predpca_{seed}.csv", data_file, delimiter=",")

    Omega = np.corrcoef(x, ui_train)[:Nv, Nv:]  # correlation matrix of x and ui (Nv, Nv)
    # binarize the matrix
    Omega_bin = np.zeros_like(Omega)
    max_idx = np.argmax(np.abs(Omega), axis=1)
    Omega_bin[np.arange(Nv), max_idx] = np.sign(Omega[np.arange(Nv), max_idx])

    ui_train = Omega_bin @ ui_train  # (Nv, T_train)
    # ui_test = Omega_bin @ ui_test  # (Nv, T_test)
    v_test = Omega_bin @ v_test  # (Nv, T_test)

    # winner-takes-all prediction
    first = slice(60, 70)
    last = slice(60 + 99990, 60 + 100000)

    img = digit_image(input_test[:, first] + input_mean)
    Image.fromarray(img).save(out_dir / f"output_{seq_label}_true_{seed}_1_10.png")
    img = digit_image(input_test[:, last] + input_mean)
    Image.fromarray(img).save(out_dir / f"output_{seq_label}_true_{seed}_99991_100000.png")

    AICs = []
    model_orders = range(1, max_model_order + 1)
    for order in model_orders:
        print(f"order: {order}")
        output, _, _, matB, AIC = wta_prediction(ui_train, v_test, input_train, input_mean, prior_u, prior_u_, order)
        AICs.append(AIC)
        img = digit_image(output[:, first] * 1.2)
        Image.fromarray(img).save(out_dir / f"output_{seq_label}_predpca{order}_{seed}_1_10.png")
        img = digit_image(output[:, last] * 1.2)
        Image.fromarray(img).save(out_dir / f"output_{seq_label}_predpca{order}_{seed}_99991_100000.png")
        np.savetxt(
            out_dir / f"output_B_{seq_label}_predpca{order}_{seed}.csv",
            np.vstack((np.arange(matB.shape[1]) + 1, matB)),
            delimiter=",",
        )

    np.savetxt(
        out_dir / f"output_AIC_{seq_label}_predpca_{seed}.csv",
        np.vstack((model_orders, AICs)),
        delimiter=",",
    )

    return AICs


if __name__ == "__main__":
    main(
        sequence_type=1,
        data_dir=data_dir,
        out_dir=out_dir,
        seed=0,
    )
