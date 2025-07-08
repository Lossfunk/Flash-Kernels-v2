import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from new_kernels.layer_norm.layer_norm import LayerNormFunction
from new_kernels.layer_norm.Functional.layer_norm import LayerNorm
from utils.utils import infer_device

device = infer_device()


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 8, 64),
        (4, 16, 128),
        (1, 1, 1023),  # Minimal batch/seq with near power-of-2 hidden
        (3, 7, 256),  # Prime numbers for batch/seq
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_liger_layer_norm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Test basic layer norm functionality against PyTorch implementation."""
    torch.manual_seed(0)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    liger_x = x.clone().requires_grad_(True)
    torch_x = x.clone().requires_grad_(True)

    liger_ln = LayerNorm(hidden_size, eps=1e-6).to(dtype).to(device)
    torch_ln = torch.nn.LayerNorm(hidden_size, eps=1e-6).to(dtype).to(device)

    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(liger_x)
    torch_output = torch_ln(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)

    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol)