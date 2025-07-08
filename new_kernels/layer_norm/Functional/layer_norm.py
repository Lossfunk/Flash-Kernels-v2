import torch
import torch.nn as nn
import torch.nn.functional as F

from new_kernels.layer_norm.layer_norm import LayerNormFunction


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, bias=False, init_fn="ones"):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size) if bias else torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Applies LayerNorm.

        We dispatch to two different implementations depending on the execution
        device:

        1. CUDA: use the custom kernel for maximum performance.
        2. CPU / other devices: fall back to PyTorch’s reference implementation
           which is much faster to import (no compilation) and sufficiently
           performant for unit-tests.
        """
        if hidden_states.is_cuda:
            # Use the high-performance fused CUDA kernel + autograd wrapper
            return LayerNormFunction.apply(hidden_states, self.weight, self.bias, self.variance_epsilon)

        # Fast fallback – relies on native PyTorch autograd
        return F.layer_norm(
            hidden_states,
            normalized_shape=(self.hidden_size,),
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        )

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"