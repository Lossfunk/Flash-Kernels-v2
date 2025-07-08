import torch
import torch.nn as nn
import torch.nn.functional as F

from new_kernels.softmax.softmax import SoftmaxFunction


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Applies Softmax along the last dimension.

        We dispatch to two different implementations depending on the execution
        device:

        1. CUDA: use the custom kernel for maximum performance.
        2. CPU / other devices: fall back to PyTorch's reference implementation
           which is much faster to import (no compilation) and sufficiently
           performant for unit-tests.
        """
        if x.is_cuda:
            # Use the high-performance fused CUDA kernel + autograd wrapper
            return SoftmaxFunction.apply(x)

        # Fast fallback – relies on native PyTorch autograd (always last dim)
        return F.softmax(x, dim=-1)
