from pathlib import Path

import numpy as np

from predpca.nonlinear.main import main

sequence_type = 2

out_dir = Path(__file__).parent / "nonlinear_main/output"
correct_file = Path(__file__).parent / f"nonlinear_main/correct/output_type{sequence_type}.npz"


def test_nonlinear_main():
    (
        qx_train,
        qx_test,
        qpsi_train,
        qpsi_test,
        B,
        qB,
        Cp,
        Lp,
    ) = main(sequence_type, out_dir)

    npz = np.load(correct_file)
    atol = 0
    np.testing.assert_allclose(qx_train, npz["qx_train"], atol=atol)
    np.testing.assert_allclose(qx_test, npz["qx_test"], atol=atol)
    np.testing.assert_allclose(qpsi_train, npz["qpsi_train"], atol=atol)
    np.testing.assert_allclose(qpsi_test, npz["qpsi_test"], atol=atol)
    np.testing.assert_allclose(B, npz["B"], atol=atol)
    np.testing.assert_allclose(qB, npz["qB"], atol=atol)
    np.testing.assert_allclose(Cp, npz["Cp"], atol=atol)
    np.testing.assert_allclose(Lp, npz["Lp"], atol=atol)


def save_correct():
    (
        qx_train,
        qx_test,
        qpsi_train,
        qpsi_test,
        B,
        qB,
        Cp,
        Lp,
    ) = main(sequence_type, out_dir=correct_file.parent)
    np.savez_compressed(
        correct_file,
        qx_train=qx_train,
        qx_test=qx_test,
        qpsi_train=qpsi_train,
        qpsi_test=qpsi_test,
        B=B,
        qB=qB,
        Cp=Cp,
        Lp=Lp,
    )


if __name__ == "__main__":
    # save_correct()
    test_nonlinear_main()
    print("All tests passed!")
