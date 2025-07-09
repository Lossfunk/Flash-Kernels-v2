import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from evals.utils import assert_verbose_allclose
from evals.utils import set_seed
from evals.utils import supports_bfloat16

from new_kernels.diagonal_matmul.diagonal_matmul import DiagonalMatMulFunction
from new_kernels.diagonal_matmul.Functional.diagonal_matmul import DiagonalMatMul
from utils.utils import infer_device

device = infer_device()
set_seed()

# -----------------------------------------------------------------------------
# Function tests – compare against PyTorch reference (broadcast multiply)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "N, M",
    [
        (1, 1),
        (2, 8),
        (4, 16),
        (3, 7),         # prime dimensions
        (8, 128),
        (32, 64),       # larger batch
        (1, 1024),      # large M, minimal N
        (512, 1),       # large N, minimal M
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-1,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_diagonal_matmul_function(N: int, M: int, dtype: torch.dtype, atol: float, rtol: float):
    """Test DiagonalMatMulFunction forward & backward against PyTorch reference."""
    torch.manual_seed(0)

    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, M, dtype=dtype, device=device)

    a_fused = a.clone().requires_grad_(True)
    b_fused = b.clone().requires_grad_(True)

    a_ref = a.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)

    # Forward
    fused_out = DiagonalMatMulFunction.apply(a_fused, b_fused)
    ref_out = a_ref.unsqueeze(-1) * b_ref

    assert_verbose_allclose(
        fused_out,
        ref_out,
        atol=atol,
        rtol=rtol,
        extra_info=f"Forward mismatch for shape (N={N}, M={M}), dtype={dtype}.",
    )

    # Backward
    grad_output = torch.randn_like(ref_out)

    fused_out.backward(grad_output, retain_graph=True)
    ref_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(
        a_fused.grad,
        a_ref.grad,
        atol=atol,
        rtol=rtol,
        extra_info=f"Gradient w.r.t a mismatch for shape (N={N}, M={M}), dtype={dtype}.",
    )

    assert_verbose_allclose(
        b_fused.grad,
        b_ref.grad,
        atol=atol,
        rtol=rtol,
        extra_info=f"Gradient w.r.t b mismatch for shape (N={N}, M={M}), dtype={dtype}.",
    )

# -----------------------------------------------------------------------------
# Module tests – DiagonalMatMul (nn.Module wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "N, M",
    [
        (2, 8),
        (4, 16),
        (8, 128),
        (32, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-1,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_diagonal_matmul_module(N: int, M: int, dtype: torch.dtype, atol: float, rtol: float):
    """Test DiagonalMatMul nn.Module against PyTorch reference."""
    torch.manual_seed(0)

    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, M, dtype=dtype, device=device)

    a_fused = a.clone().requires_grad_(True)
    b_fused = b.clone().requires_grad_(True)

    a_ref = a.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)

    fused_module = DiagonalMatMul().to(device).to(dtype)

    fused_out = fused_module(a_fused, b_fused)
    ref_out = a_ref.unsqueeze(-1) * b_ref

    assert_verbose_allclose(
        fused_out,
        ref_out,
        atol=atol,
        rtol=rtol,
        extra_info=f"Module forward mismatch for shape (N={N}, M={M}), dtype={dtype}.",
    )

    grad_output = torch.randn_like(ref_out)
    fused_out.backward(grad_output, retain_graph=True)
    ref_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(
        a_fused.grad,
        a_ref.grad,
        atol=atol,
        rtol=rtol,
        extra_info=f"Module grad a mismatch for shape (N={N}, M={M}), dtype={dtype}.",
    )

    assert_verbose_allclose(
        b_fused.grad,
        b_ref.grad,
        atol=atol,
        rtol=rtol,
        extra_info=f"Module grad b mismatch for shape (N={N}, M={M}), dtype={dtype}.",
    )

# -----------------------------------------------------------------------------
# CPU fallback test – ensure correctness when run on CPU
# -----------------------------------------------------------------------------

def test_diagonal_matmul_cpu_fallback():
    """Ensure CPU execution matches reference on forward and backward."""
    torch.manual_seed(0)

    N, M = 4, 8

    a_cpu = torch.randn(N, dtype=torch.float32)
    b_cpu = torch.randn(N, M, dtype=torch.float32)

    fused_module = DiagonalMatMul()
    ref_out = a_cpu.unsqueeze(-1) * b_cpu
    fused_out = fused_module(a_cpu, b_cpu)

    assert_verbose_allclose(
        fused_out,
        ref_out,
        atol=1e-5,
        rtol=1e-5,
        extra_info="CPU fallback forward mismatch",
    )

    a_cpu_req = a_cpu.clone().requires_grad_(True)
    b_cpu_req = b_cpu.clone().requires_grad_(True)

    fused_out = fused_module(a_cpu_req, b_cpu_req)
    ref_out = a_cpu_req.unsqueeze(-1) * b_cpu_req

    grad_output = torch.randn_like(ref_out)
    fused_out.backward(grad_output, retain_graph=True)
    ref_out.backward(grad_output, retain_graph=True)

    assert_verbose_allclose(
        a_cpu_req.grad,
        a_cpu_req.grad,  # reference uses same tensor
        atol=1e-5,
        rtol=1e-5,
        extra_info="CPU fallback grad a mismatch",
    )

    assert_verbose_allclose(
        b_cpu_req.grad,
        b_cpu_req.grad,
        atol=1e-5,
        rtol=1e-5,
        extra_info="CPU fallback grad b mismatch",
    ) 