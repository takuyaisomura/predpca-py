import torch


def sym_positive(A: torch.Tensor) -> torch.Tensor:
    A_sym = (A + A.T) / 2
    min_eig = torch.linalg.eigh(A_sym)[0].min()
    if min_eig < 0:
        A_sym -= torch.eye(A_sym.shape[0], dtype=A_sym.dtype, device=A_sym.device) * min_eig
    return A_sym
