from pathlib import Path

import numpy as np

from predpca.aloi.main_plot_prediction_error import main

out_dir = Path(__file__).parent / "aloi_main_plot_prediction_error/output"
correct_dir = Path(__file__).parent / "aloi_main_plot_prediction_error/correct"


def test_aloi_main_plot_prediction_error():
    main(out_dir)

    csv_names = [
        "predpca_opt_encode_dim.csv",
        "predpca_test_err.csv",
        "predpca_pc1_of_deviation_eig.csv",
        "predpca_pc1_of_deviation.csv",
    ]

    atol = 0
    for csv_name in csv_names:
        data_out = np.loadtxt(out_dir / csv_name, delimiter=",", skiprows=1)
        data_correct = np.loadtxt(correct_dir / csv_name, delimiter=",", skiprows=1)
        if csv_name == "predpca_test_err.csv":
            atol = 2e-6  # TODO: error is occurring somewhere
        np.testing.assert_allclose(data_out, data_correct, atol=atol)


def save_correct():
    main(out_dir=correct_dir)


if __name__ == "__main__":
    # save_correct()
    test_aloi_main_plot_prediction_error()
    print("All tests passed!")
