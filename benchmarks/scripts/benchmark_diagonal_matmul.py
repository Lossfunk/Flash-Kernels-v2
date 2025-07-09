"""
python benchmark_diagonal_matmul.py --overwrite

Benchmarks diagonal-matrix multiply (C = diag(a) @ B) for speed and memory across
various dimensions. Results are stored via utils.run_benchmarks for later
visualisation.
"""

import torch
import torch.utils.benchmark as benchmark
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

def _extract_quantiles_from_measurement(measurement, quantiles: list[float] = [0.5, 0.2, 0.8]):
    """Extract selected quantiles (in ms) from a torch.utils.benchmark.Measurement."""
    # Convert seconds to milliseconds
    times_ms = [t * 1000 for t in measurement.raw_times]
    times_tensor = torch.tensor(times_ms, dtype=torch.float32)

    quantile_values = torch.quantile(times_tensor, torch.tensor(quantiles, dtype=torch.float32))
    return (
        quantile_values[0].item(),
        quantile_values[1].item(),
        quantile_values[2].item(),
    )


def _run_pytorch_benchmark(stmt_fn, globals_dict, label: str, description: str):
    """Utility that wraps torch.utils.benchmark for adaptive timing benchmarking."""
    timer = benchmark.Timer(
        stmt="stmt_fn()",
        globals=globals_dict,
        label=label,
        description=description,
        num_threads=1,  # single-threaded for reproducible results
    )

    measurement = timer.blocked_autorange()
    return _extract_quantiles_from_measurement(measurement, QUANTILES)


# -----------------------------------------------------------------------------
# SPEED BENCHMARKS
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
        if provider == "liger" or provider == "custom":
            return diag_mm(a, B)
        if provider == "torch":
            return a.unsqueeze(-1) * B

    # Warm-up
    for _ in range(5):
        y_fwd()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if mode == "forward":
        ms_50, ms_20, ms_80 = _run_pytorch_benchmark(
            y_fwd,
            {"stmt_fn": y_fwd, "a": a, "B": B},
            label=f"DiagMatMul Forward ({provider})",
            description=f"N={N}, M={M}, dtype={dtype}",
        )
    elif mode == "backward":
        y = y_fwd()
        # warm-up backward
        for _ in range(5):
            y_fwd().backward(dy, retain_graph=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        backward_fn = lambda: y.backward(dy, retain_graph=True)
        ms_50, ms_20, ms_80 = _run_pytorch_benchmark(
            backward_fn,
            {"stmt_fn": backward_fn, "y": y, "dy": dy},
            label=f"DiagMatMul Backward ({provider})",
            description=f"N={N}, M={M}, dtype={dtype}",
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

        ms_50, ms_20, ms_80 = _run_pytorch_benchmark(
            full,
            {"stmt_fn": full, "a": a, "B": B, "dy": dy},
            label=f"DiagMatMul Full ({provider})",
            description=f"N={N}, M={M}, dtype={dtype}",
        )
    else:
        raise ValueError(f"Unknown operation mode: {mode}")

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
        if provider == "liger" or provider == "custom":
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
        kernel_providers=["liger", "torch"],
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