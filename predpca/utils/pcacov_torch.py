import torch


def pcacov(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # sort eigenvalues in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # set very small eigenvalues to machine precision
    eigenvalues = torch.clamp(eigenvalues, min=torch.finfo(cov.dtype).eps)

    # adjust eigenvectors so that the maximum absolute value element of each column is positive
    max_abs_idx = torch.argmax(torch.abs(eigenvectors), dim=0)
    signs = torch.sign(eigenvectors[max_abs_idx, torch.arange(eigenvectors.shape[1])])
    eigenvectors *= signs

    # calculate contribution ratio
    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    return eigenvectors, eigenvalues, explained_variance_ratio
