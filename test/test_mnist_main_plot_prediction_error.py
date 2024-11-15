from pathlib import Path

import numpy as np

from predpca.mnist import main_plot_prediction_error

sequence_type = 1
seed = 0
data_dir = Path(main_plot_prediction_error.__file__).parent / "data"
out_dir = Path(__file__).parent / "mnist_main_plot_prediction_error/output"
correct_file = Path(__file__).parent / "mnist_main_plot_prediction_error/correct/output.npz"


def test_mnist_main_plot_prediction_error():
    err_param, err_optimal, err_predpca, err_supervised = main_plot_prediction_error.main(
        sequence_type,
        data_dir,
        out_dir,
        seed,
    )

    npz = np.load(correct_file, allow_pickle=True)
    atol = 0
    np.testing.assert_allclose(err_param, npz["err_param"], atol=atol)
    np.testing.assert_allclose(err_optimal, npz["err_optimal"], atol=atol)
    np.testing.assert_allclose(err_predpca["th"], npz["err_predpca"].item()["th"], atol=atol)
    np.testing.assert_allclose(err_predpca["em"], npz["err_predpca"].item()["em"], atol=atol)
    np.testing.assert_allclose(err_supervised["th"], npz["err_supervised"].item()["th"], atol=atol)
    np.testing.assert_allclose(err_supervised["em"], npz["err_supervised"].item()["em"], atol=atol)


def save_correct():
    err_param, err_optimal, err_predpca, err_supervised = main_plot_prediction_error.main(
        sequence_type,
        data_dir,
        out_dir=correct_file.parent,
        seed=seed,
    )
    np.savez_compressed(
        correct_file,
        err_param=err_param,
        err_optimal=err_optimal,
        err_predpca=err_predpca,
        err_supervised=err_supervised,
    )


if __name__ == "__main__":
    # save_correct()
    test_mnist_main_plot_prediction_error()
    print("All tests passed!")
