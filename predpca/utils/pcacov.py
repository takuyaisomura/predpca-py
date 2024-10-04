import numpy as np
from scipy import linalg


def pcacov(cov):
    (
        eigenvalues,  # (N,)
        eigenvectors,  # (N, N); i-th column is the eigenvector corresponding to the i-th eigenvalue
    ) = linalg.eigh(cov)

    # sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Set very small or negative eigenvalues (that are not precise) to machine precision
    eigenvalues = np.maximum(eigenvalues, np.finfo(cov.dtype).eps)

    # adjust eigenvectors so that the maximum absolute value element of each column is positive
    max_abs_idx = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[max_abs_idx, range(eigenvectors.shape[1])])
    eigenvectors *= signs

    # calculate contribution ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    return eigenvectors, eigenvalues, explained_variance_ratio
