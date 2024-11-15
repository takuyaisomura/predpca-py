from pathlib import Path

import numpy as np

from predpca.mnist import main_plot_encodings

sequence_type = 1

data_dir = Path(main_plot_encodings.__file__).parent / "data"
out_dir = Path(__file__).parent / "mnist_main_plot_encodings/output"
correct_file = Path(__file__).parent / "mnist_main_plot_encodings/correct/output.npz"


def test_mnist_main_plot_encodings():
    Wpca, Wppca, Wica, A = main_plot_encodings.main(sequence_type, data_dir, out_dir)

    npz = np.load(correct_file)
    atol = 0
    np.testing.assert_allclose(Wpca, npz["Wpca"], atol=atol)
    np.testing.assert_allclose(Wppca, npz["Wppca"], atol=atol)
    np.testing.assert_allclose(Wica, npz["Wica"], atol=atol)
    np.testing.assert_allclose(A, npz["A"], atol=atol)


def save_correct():
    Wpca, Wppca, Wica, A = main_plot_encodings.main(sequence_type, data_dir, out_dir=correct_file.parent)
    np.savez_compressed(correct_file, Wpca=Wpca, Wppca=Wppca, Wica=Wica, A=A)


if __name__ == "__main__":
    # save_correct()
    test_mnist_main_plot_encodings()
    print("All tests passed!")
