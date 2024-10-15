from pathlib import Path

import numpy as np

from predpca.mnist import main_plot_predicted_images

seed = 0
sequence_type = 1
# sequence_type = 2
max_model_order = 2

data_dir = Path(main_plot_predicted_images.__file__).parent / "data"
out_dir = Path(__file__).parent / "mnist_main_plot_predicted_images/output"
correct_file = Path(__file__).parent / "mnist_main_plot_predicted_images/correct/output.npz"


def test_mnist_main_plot_predicted_images():
    AICs = main_plot_predicted_images.main(
        sequence_type,
        data_dir,
        out_dir,
        max_model_order,
        seed,
    )

    npz = np.load(correct_file)
    atol = 0
    np.testing.assert_allclose(AICs, npz["AICs"][:max_model_order], atol=atol)


def save_correct():
    AICs = main_plot_predicted_images.main(
        sequence_type,
        data_dir,
        out_dir=correct_file.parent,
        seed=seed,
    )
    np.savez_compressed(correct_file, AICs=AICs)


if __name__ == "__main__":
    # save_correct()
    test_mnist_main_plot_predicted_images()
    print("All tests passed!")
