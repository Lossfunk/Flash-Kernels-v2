"""
python benchmark_diagonal_matmul.py --overwrite

Benchmarks diagonal-matrix multiply (C = diag(a) @ B) for speed and memory across
various dimensions. Results are stored via utils.run_benchmarks for later
visualisation.
"""

import torch
import triton  # Use Triton's built-in benchmarking utility
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

from new_kernels.diagonal_matmul.Functional.diagonal_matmul import DiagonalMatMul

device = infer_device()

# -----------------------------------------------------------------------------
# Replaced torch.utils.benchmark utilities with Triton's do_bench for timing.
# -----------------------------------------------------------------------------

def bench_speed_diag_matmul(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark speed (ms) for DiagonalMatMul forward/backward/full passes."""
    N = input.x  # number of rows / batch size
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra = input.extra_benchmark_config or {}
    M = extra.get("M", 2048)
    dtype = extra.get("dtype", torch.float32)

    # Shapes: a: [N], B: [N, M]
    a = torch.randn(N, dtype=dtype, device=device)
    B = torch.randn(N, M, dtype=dtype, device=device)
    dy = torch.randn(N, M, dtype=dtype, device=device)

    a.requires_grad_(True)
    B.requires_grad_(True)

    diag_mm = DiagonalMatMul().to(device)

    def y_fwd():
        if provider == "custom":
            return diag_mm(a, B)
        if provider == "torch":
            return a.unsqueeze(-1) * B

    # Warm-up
    for _ in range(5):
        y_fwd()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            quantiles=QUANTILES,
            grad_to_none=[a, B],
            rep=500,
        )
    elif mode == "backward":
        y = y_fwd()
        # warm-up backward
        for _ in range(5):
            y_fwd().backward(dy, retain_graph=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[a, B],
            rep=500,
        )
    elif mode == "full":
        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        # warm-up full pass
        for _ in range(5):
            full()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            quantiles=QUANTILES,
            grad_to_none=[a, B],
            rep=500,
        )
    else:
        raise ValueError(f"Unknown operation mode: {mode}")

    # Fail fast if Triton produced None timings
    if any(val is None for val in (ms_20, ms_50, ms_80)):
        raise RuntimeError(
            f"Benchmark speed result is None: ms_20={ms_20}, ms_50={ms_50}, ms_80={ms_80}"
        )

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


# -----------------------------------------------------------------------------
# MEMORY BENCHMARKS
# -----------------------------------------------------------------------------

def bench_memory_diag_matmul(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark memory footprint (MB) of full forward+backward diag matmul pass."""
    N = input.x
    provider = input.kernel_provider
    extra = input.extra_benchmark_config or {}
    dtype = extra.get("dtype", torch.float32)
    M = extra.get("M", 2048)

    a = torch.randn(N, dtype=dtype, device=device)
    B = torch.randn(N, M, dtype=dtype, device=device)
    dy = torch.randn(N, M, dtype=dtype, device=device)

    a.requires_grad_(True)
    B.requires_grad_(True)

    diag_mm = DiagonalMatMul().to(device)

    def y_fwd():
        if provider == "custom":
            return diag_mm(a, B)
        if provider == "torch":
            return a.unsqueeze(-1) * B

    def full_pass():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full_pass, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = dict(
        kernel_name="diag_matmul",
        x_name="N",
        x_label="rows",
        x_values=[128, 256, 512, 1024, 2048, 4096],
        kernel_providers=["custom", "torch"],
        extra_benchmark_configs=[
            {"M": 2048, "dtype": torch.float32},
            {"M": 2048, "dtype": torch.bfloat16},
        ],
    )

    # Speed (ms)
    run_benchmarks(
        bench_test_fn=bench_speed_diag_matmul,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        overwrite=args.overwrite,
        **common_configs,
    )

    # Memory (MB)
    run_benchmarks(
        bench_test_fn=bench_memory_diag_matmul,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        overwrite=args.overwrite,
        **common_configs,
    ) 