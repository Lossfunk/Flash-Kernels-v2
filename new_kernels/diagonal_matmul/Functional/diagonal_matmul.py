import torch
import torch.nn as nn

from new_kernels.diagonal_matmul.diagonal_matmul import DiagonalMatMulFunction


class DiagonalMatMul(nn.Module):
    """Module wrapper for diagonal matrix multiplication.

    Computes out = diag(a) @ b where a.shape == (N,) and b.shape == (N, M).
    """

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.is_cuda:
            return DiagonalMatMulFunction.apply(a, b)
        # fallback (CPU) – rely on broadcast multiply and autograd
        return a.unsqueeze(-1) * b 