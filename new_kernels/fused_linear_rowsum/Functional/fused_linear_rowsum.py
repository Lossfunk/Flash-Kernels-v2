import torch
import torch.nn as nn
import torch.nn.functional as F

from new_kernels.fused_linear_rowsum.fused_linear_rowsum import FusedLinearRowSumFunction


class FusedLinearRowSum(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        
        # Initialize bias vector
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies fused linear transformation followed by row-wise summation.

        Computes: sum((x @ weight.T) + bias, dim=-1)

        We dispatch to two different implementations depending on the execution
        device:

        1. CUDA: use the custom fused kernel for maximum performance.
        2. CPU / other devices: fall back to PyTorch's reference implementation
           which is much faster to import (no compilation) and sufficiently
           performant for unit-tests.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size] containing row-wise sums
        """
        if x.is_cuda:
            # Use the high-performance fused CUDA kernel + autograd wrapper
            return FusedLinearRowSumFunction.apply(x, self.weight, self.bias)

        # Fast fallback – relies on native PyTorch autograd
        linear_out = F.linear(x, self.weight, self.bias)
        return torch.sum(linear_out, dim=-1)

    def extra_repr(self):
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias is not None}' 