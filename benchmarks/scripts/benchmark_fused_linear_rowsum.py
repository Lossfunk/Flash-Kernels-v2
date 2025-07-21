import torch
import triton
import sys
import os

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.utils import (
    QUANTILES,
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
    infer_device,
)

from new_kernels.fused_linear_rowsum.Functional.fused_linear_rowsum import FusedLinearRowSum

device = infer_device()


# -----------------------------------------------------------------------------
# Replaced torch.utils.benchmark utilities with Triton's do_bench for timing.
# -----------------------------------------------------------------------------


def bench_speed_fused_linear_rowsum(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark speed (ms) for FusedLinearRowSum forward/backward/full passes."""
    input_dim = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra = input.extra_benchmark_config or {}
    batch_size = extra.get("batch_size", 1024)
    output_dim = extra.get("output_dim", 2048)
    dtype = extra.get("dtype", torch.float32)
    bias = extra.get("bias", True)

    # Create models with proper dtype
    custom_fused_linear = FusedLinearRowSum(input_dim, output_dim, bias=bias).to(device, dtype=dtype)
    torch_linear = torch.nn.Linear(input_dim, output_dim, bias=bias).to(device, dtype=dtype)

    # Create input data
    x = torch.randn(batch_size, input_dim, dtype=dtype, device=device)
    dy = torch.randn(batch_size, dtype=dtype, device=device)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "custom":
            return custom_fused_linear(x)
        if provider == "torch":
            # PyTorch reference: linear + sum
            linear_out = torch_linear(x)
            return torch.sum(linear_out, dim=-1)

    # Warm-up to stabilise GPU clocks
    for _ in range(5):
        y_fwd()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "backward":
        y = y_fwd()
        # Warm-up backward
        for _ in range(5):
            y_fwd().backward(dy, retain_graph=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    elif mode == "full":
        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        # Warm-up full pass
        for _ in range(5):
            full()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    else:
        raise ValueError(f"Unknown kernel operation mode: {mode}")

    # Fail fast if Triton produced None timings
    if any(val is None for val in (ms_20, ms_50, ms_80)):
        raise RuntimeError(
            f"Benchmark speed result is None: ms_20={ms_20}, ms_50={ms_50}, ms_80={ms_80}"
        )

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_fused_linear_rowsum(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark memory footprint (MB) of a full FusedLinearRowSum forward+backward pass."""
    input_dim = input.x
    provider = input.kernel_provider
    extra = input.extra_benchmark_config or {}
    dtype = extra.get("dtype", torch.float32)
    batch_size = extra.get("batch_size", 1024)
    output_dim = extra.get("output_dim", 2048)
    bias = extra.get("bias", True)

    # Create models with proper dtype
    custom_fused_linear = FusedLinearRowSum(input_dim, output_dim, bias=bias).to(device, dtype=dtype)
    torch_linear = torch.nn.Linear(input_dim, output_dim, bias=bias).to(device, dtype=dtype)

    # Create input data
    x = torch.randn(batch_size, input_dim, dtype=dtype, device=device)
    dy = torch.randn(batch_size, dtype=dtype, device=device)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "custom":
            return custom_fused_linear(x)
        if provider == "torch":
            # PyTorch reference: linear + sum
            linear_out = torch_linear(x)
            return torch.sum(linear_out, dim=-1)

    def full_pass():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full_pass, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = dict(
        kernel_name="fused_linear_rowsum",
        x_name="input_dim",
        x_label="input dimension",
        x_values=[512, 1024, 2048, 4096, 8192],
        kernel_providers=["custom", "torch"],
        extra_benchmark_configs=[
            {"batch_size": 1024, "output_dim": 2048, "dtype": torch.float32, "bias": True},
            {"batch_size": 1024, "output_dim": 2048, "dtype": torch.bfloat16, "bias": True},
            {"batch_size": 512, "output_dim": 4096, "dtype": torch.float32, "bias": True},
            {"batch_size": 2048, "output_dim": 1024, "dtype": torch.float32, "bias": True},
        ],
    )

    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_rowsum,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        overwrite=args.overwrite,
        **common_configs,
    )

    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_rowsum,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        overwrite=args.overwrite,
        **common_configs,
    ) 