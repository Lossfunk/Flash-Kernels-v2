import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.utils import assert_verbose_allclose
from evals.utils import set_seed
from evals.utils import supports_bfloat16

from new_kernels.fused_linear_rowsum.fused_linear_rowsum import FusedLinearRowSumFunction
from new_kernels.fused_linear_rowsum.Functional.fused_linear_rowsum import FusedLinearRowSum
from utils.utils import infer_device

device = infer_device()
set_seed()


@pytest.mark.parametrize(
    "batch_size, input_dim, output_dim",
    [
        (2, 8, 4),
        (4, 16, 8),
        (1, 1, 1),  # Minimal case
        (3, 7, 5),  # Prime numbers
        (8, 128, 64),  # Larger dimensions
        (16, 512, 256),  # Even larger
        (1, 1023, 512),  # Large input dimension
        (32, 64, 1024),  # Large output dimension
    ],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 3e-5, 3e-5),
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
def test_fused_linear_rowsum_function(
    batch_size: int,
    input_dim: int,
    output_dim: int,
    bias: bool,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Test FusedLinearRowSumFunction against PyTorch reference implementation."""
    torch.manual_seed(0)

    # Create test data
    x = torch.randn(batch_size, input_dim, dtype=dtype, device=device)
    weight = torch.randn(output_dim, input_dim, dtype=dtype, device=device)
    if bias:
        bias_tensor = torch.randn(output_dim, dtype=dtype, device=device)
    else:
        bias_tensor = None

    # Test inputs for gradient computation
    x_fused = x.clone().requires_grad_(True)
    weight_fused = weight.clone().requires_grad_(True)
    bias_fused = bias_tensor.clone().requires_grad_(True) if bias_tensor is not None else None

    x_ref = x.clone().requires_grad_(True)
    weight_ref = weight.clone().requires_grad_(True)
    bias_ref = bias_tensor.clone().requires_grad_(True) if bias_tensor is not None else None

    # Fused implementation
    fused_output = FusedLinearRowSumFunction.apply(x_fused, weight_fused, bias_fused)

    # Reference implementation: Linear + RowSum
    linear_out = F.linear(x_ref, weight_ref, bias_ref)
    ref_output = torch.sum(linear_out, dim=-1)

    # Test forward pass
    assert_verbose_allclose(
        fused_output, 
        ref_output, 
        atol=atol, 
        rtol=rtol,
        extra_info=f"Forward pass failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
    )

    # Test backward pass
    grad_output = torch.randn(batch_size, dtype=dtype, device=device)
    
    fused_output.backward(grad_output, retain_graph=True)
    ref_output.backward(grad_output, retain_graph=True)

    # Check input gradients
    assert_verbose_allclose(
        x_fused.grad, 
        x_ref.grad, 
        atol=atol, 
        rtol=rtol,
        extra_info=f"Input gradient failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
    )

    # Check weight gradients
    assert_verbose_allclose(
        weight_fused.grad, 
        weight_ref.grad, 
        atol=atol, 
        rtol=rtol,
        extra_info=f"Weight gradient failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
    )

    # Check bias gradients (if bias is used)
    if bias:
        assert_verbose_allclose(
            bias_fused.grad, 
            bias_ref.grad, 
            atol=atol, 
            rtol=rtol,
            extra_info=f"Bias gradient failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
        )


@pytest.mark.parametrize(
    "batch_size, input_dim, output_dim",
    [
        (2, 8, 4),
        (4, 16, 8),
        (1, 1, 1),
        (3, 7, 5),
        (8, 128, 64),
        (16, 512, 256),
        (1, 1023, 512),
        (32, 64, 1024),
    ],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 3e-5, 3e-5),
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
def test_fused_linear_rowsum_module(
    batch_size: int,
    input_dim: int,
    output_dim: int,
    bias: bool,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Test FusedLinearRowSum module against PyTorch reference implementation."""
    torch.manual_seed(0)

    # Create test data
    x = torch.randn(batch_size, input_dim, dtype=dtype, device=device)
    
    # Test inputs for gradient computation
    x_fused = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    # Create modules
    fused_layer = FusedLinearRowSum(input_dim, output_dim, bias=bias).to(dtype).to(device)
    
    # Reference implementation using separate Linear + sum
    ref_linear = nn.Linear(input_dim, output_dim, bias=bias).to(dtype).to(device)
    
    # Copy weights to ensure fair comparison
    with torch.no_grad():
        ref_linear.weight.copy_(fused_layer.weight)
        if bias:
            ref_linear.bias.copy_(fused_layer.bias)

    # Forward pass
    fused_output = fused_layer(x_fused)
    ref_linear_out = ref_linear(x_ref)
    ref_output = torch.sum(ref_linear_out, dim=-1)

    # Test forward pass
    assert_verbose_allclose(
        fused_output, 
        ref_output, 
        atol=atol, 
        rtol=rtol,
        extra_info=f"Module forward pass failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
    )

    # Test backward pass
    grad_output = torch.randn(batch_size, dtype=dtype, device=device)
    
    fused_output.backward(grad_output, retain_graph=True)
    ref_output.backward(grad_output, retain_graph=True)

    # Check input gradients
    assert_verbose_allclose(
        x_fused.grad, 
        x_ref.grad, 
        atol=atol, 
        rtol=rtol,
        extra_info=f"Module input gradient failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
    )

    # Check weight gradients
    assert_verbose_allclose(
        fused_layer.weight.grad, 
        ref_linear.weight.grad, 
        atol=atol, 
        rtol=rtol,
        extra_info=f"Module weight gradient failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
    )

    # Check bias gradients (if bias is used)
    if bias:
        assert_verbose_allclose(
            fused_layer.bias.grad, 
            ref_linear.bias.grad, 
            atol=atol, 
            rtol=rtol,
            extra_info=f"Module bias gradient failed for shape ({batch_size}, {input_dim}, {output_dim}), bias={bias}, dtype={dtype}"
        )


@pytest.mark.parametrize(
    "batch_size, input_dim, output_dim",
    [
        (1, 1, 1),  # Edge case: minimal dimensions
        (1, 2048, 1),  # Large input, minimal output
        (1, 1, 2048),  # Minimal input, large output
        (1024, 1, 1),  # Large batch, minimal dimensions
    ],
)
def test_fused_linear_rowsum_edge_cases(
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> None:
    """Test edge cases for FusedLinearRowSum."""
    torch.manual_seed(0)
    
    x = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
    weight = torch.randn(output_dim, input_dim, device=device, dtype=torch.float32)
    bias = torch.randn(output_dim, device=device, dtype=torch.float32)
    
    # Test with bias
    output_with_bias = FusedLinearRowSumFunction.apply(x, weight, bias)
    ref_with_bias = torch.sum(F.linear(x, weight, bias), dim=-1)
    
    assert_verbose_allclose(
        output_with_bias, 
        ref_with_bias, 
        atol=3e-5, 
        rtol=3e-5,
        extra_info=f"Edge case with bias failed for shape ({batch_size}, {input_dim}, {output_dim})"
    )
    
    # Test without bias
    output_no_bias = FusedLinearRowSumFunction.apply(x, weight, None)
    ref_no_bias = torch.sum(F.linear(x, weight, None), dim=-1)
    
    assert_verbose_allclose(
        output_no_bias, 
        ref_no_bias, 
        atol=3e-5, 
        rtol=3e-5,
        extra_info=f"Edge case without bias failed for shape ({batch_size}, {input_dim}, {output_dim})"
    )


def test_fused_linear_rowsum_cpu_fallback():
    """Test that CPU fallback works correctly."""
    torch.manual_seed(0)
    
    batch_size, input_dim, output_dim = 4, 8, 6
    
    # Create CPU tensors
    x_cpu = torch.randn(batch_size, input_dim, dtype=torch.float32)
    
    # Create module on CPU
    fused_layer_cpu = FusedLinearRowSum(input_dim, output_dim, bias=True)
    ref_linear_cpu = nn.Linear(input_dim, output_dim, bias=True)
    
    # Copy weights
    with torch.no_grad():
        ref_linear_cpu.weight.copy_(fused_layer_cpu.weight)
        ref_linear_cpu.bias.copy_(fused_layer_cpu.bias)
    
    # Test forward pass on CPU
    fused_output_cpu = fused_layer_cpu(x_cpu)
    ref_output_cpu = torch.sum(ref_linear_cpu(x_cpu), dim=-1)
    
    assert_verbose_allclose(
        fused_output_cpu, 
        ref_output_cpu, 
        atol=3e-5, 
        rtol=3e-5,
        extra_info="CPU fallback forward pass failed"
    )
    
    # Test backward pass on CPU
    grad_output = torch.randn(batch_size)
    x_cpu_grad = x_cpu.clone().requires_grad_(True)
    x_ref_grad = x_cpu.clone().requires_grad_(True)
    
    fused_output_cpu = fused_layer_cpu(x_cpu_grad)
    ref_output_cpu = torch.sum(ref_linear_cpu(x_ref_grad), dim=-1)
    
    fused_output_cpu.backward(grad_output)
    ref_output_cpu.backward(grad_output)
    
    assert_verbose_allclose(
        x_cpu_grad.grad, 
        x_ref_grad.grad, 
        atol=3e-5, 
        rtol=3e-5,
        extra_info="CPU fallback backward pass failed"
    )


def test_fused_linear_rowsum_mathematical_properties():
    """Test mathematical properties of the fused linear rowsum operation."""
    torch.manual_seed(0)
    
    batch_size, input_dim, output_dim = 4, 6, 8
    
    x = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
    weight = torch.randn(output_dim, input_dim, device=device, dtype=torch.float32)
    bias = torch.randn(output_dim, device=device, dtype=torch.float32)
    
    # Test linearity without bias: f(ax + by) = af(x) + bf(y)
    x1 = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
    x2 = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
    a, b = 2.0, 3.0
    
    # f(ax1 + bx2) without bias
    combined_input = a * x1 + b * x2
    combined_output = FusedLinearRowSumFunction.apply(combined_input, weight, None)
    
    # af(x1) + bf(x2) without bias
    output1 = FusedLinearRowSumFunction.apply(x1, weight, None)
    output2 = FusedLinearRowSumFunction.apply(x2, weight, None)
    linear_combination = a * output1 + b * output2
    
    assert_verbose_allclose(
        combined_output, 
        linear_combination, 
        atol=3e-5, 
        rtol=3e-5,
        extra_info="Linearity property test failed (without bias)"
    )
    
    # Test zero input
    zero_input = torch.zeros(batch_size, input_dim, device=device, dtype=torch.float32)
    zero_output = FusedLinearRowSumFunction.apply(zero_input, weight, bias)
    expected_zero_output = torch.sum(bias).expand(batch_size)
    
    assert_verbose_allclose(
        zero_output, 
        expected_zero_output, 
        atol=3e-5, 
        rtol=3e-5,
        extra_info="Zero input test failed"
    )


def test_fused_linear_rowsum_module_repr():
    """Test the string representation of the FusedLinearRowSum module."""
    # Test with bias
    layer_with_bias = FusedLinearRowSum(128, 256, bias=True)
    expected_repr_with_bias = "input_dim=128, output_dim=256, bias=True"
    assert layer_with_bias.extra_repr() == expected_repr_with_bias
    
    # Test without bias
    layer_without_bias = FusedLinearRowSum(64, 128, bias=False)
    expected_repr_without_bias = "input_dim=64, output_dim=128, bias=False"
    assert layer_without_bias.extra_repr() == expected_repr_without_bias


@pytest.mark.skipif(device == "cpu", reason="Memory test requires CUDA")
def test_fused_linear_rowsum_memory_efficiency():
    """Test that fused kernel uses memory efficiently."""
    torch.manual_seed(0)
    
    batch_size, input_dim, output_dim = 1024, 512, 1024
    
    x = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
    weight = torch.randn(output_dim, input_dim, device=device, dtype=torch.float32)
    bias = torch.randn(output_dim, device=device, dtype=torch.float32)
    
    # Measure memory for fused implementation
    torch.cuda.reset_peak_memory_stats()
    fused_output = FusedLinearRowSumFunction.apply(x, weight, bias)
    fused_memory = torch.cuda.max_memory_allocated()
    
    # Measure memory for reference implementation
    torch.cuda.reset_peak_memory_stats()
    ref_linear_out = F.linear(x, weight, bias)
    ref_output = torch.sum(ref_linear_out, dim=-1)
    ref_memory = torch.cuda.max_memory_allocated()
    
    # Fused kernel should use less memory (no need to store intermediate linear output)
    print(f"Fused memory: {fused_memory / 1024**2:.2f} MB")
    print(f"Reference memory: {ref_memory / 1024**2:.2f} MB")
    print(f"Memory ratio (fused/ref): {fused_memory / ref_memory:.2f}")
    
    # The fused implementation should use less memory
    # (allowing some tolerance for CUDA memory allocation overhead)
    assert fused_memory < ref_memory * 1.1, "Fused implementation should be more memory efficient" 