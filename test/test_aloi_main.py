from pathlib import Path

import numpy as np

from predpca.aloi.main import main

out_dir = Path(__file__).parent / "aloi_main/output"
correct_file = Path(__file__).parent / "aloi_main/correct/output.npz"


def test_aloi_main():
    u_test, u_sub_ = main(out_dir)

    npz = np.load(correct_file)
    atol = 0
    np.testing.assert_allclose(u_test, npz["u_test"], atol=atol)
    np.testing.assert_allclose(u_sub_, npz["u_sub_"], atol=atol)


def save_correct():
    u_test, u_sub_ = main(out_dir=correct_file.parent)
    np.savez_compressed(correct_file, u_test=u_test, u_sub_=u_sub_)


if __name__ == "__main__":
    # save_correct()
    test_aloi_main()
    print("All tests passed!")
