"""
python benchmarks_visualizer.py \
    --kernel-name kto_loss \
    --metric-name speed \
    --kernel-operation-mode forward backward
"""

import torch
import triton
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

from new_kernels.softmax.Functional.softmax import Softmax


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


def bench_speed_softmax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark speed (ms) for Softmax forward/backward/full passes."""
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra = input.extra_benchmark_config or {}
    M = extra.get("M", 2048)
    dtype = extra.get("dtype", torch.float32)

    x_shape = (M, N)

    custom_sm = Softmax().to(device)
    torch_sm = torch.nn.Softmax(dim=-1).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger" or provider == "custom":
            return custom_sm(x)
        if provider == "torch":
            return torch_sm(x)

    # Warm-up to stabilise GPU clocks
    for _ in range(5):
        y_fwd()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if mode == "forward":
        ms_50, ms_20, ms_80 = _run_pytorch_benchmark(
            y_fwd,
            {"stmt_fn": y_fwd, "x": x},
            label=f"Softmax Forward ({provider})",
            description=f"M={M}, N={N}, dtype={dtype}",
        )
    elif mode == "backward":
        y = y_fwd()
        # Warm-up backward
        for _ in range(5):
            y_fwd().backward(dy, retain_graph=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        backward_fn = lambda: y.backward(dy, retain_graph=True)
        ms_50, ms_20, ms_80 = _run_pytorch_benchmark(
            backward_fn,
            {"stmt_fn": backward_fn, "x": x, "y": y, "dy": dy},
            label=f"Softmax Backward ({provider})",
            description=f"M={M}, N={N}, dtype={dtype}",
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

        ms_50, ms_20, ms_80 = _run_pytorch_benchmark(
            full,
            {"stmt_fn": full, "x": x, "dy": dy},
            label=f"Softmax Full ({provider})",
            description=f"M={M}, N={N}, dtype={dtype}",
        )
    else:
        raise ValueError(f"Unknown kernel operation mode: {mode}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_softmax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark memory footprint (MB) of a full Softmax forward+backward pass."""
    N = input.x
    provider = input.kernel_provider
    extra = input.extra_benchmark_config or {}
    dtype = extra.get("dtype", torch.float32)
    M = extra.get("M", 2048)

    x_shape = (M, N)

    custom_sm = Softmax().to(device)
    torch_sm = torch.nn.Softmax(dim=-1).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger" or provider == "custom":
            return custom_sm(x)
        if provider == "torch":
            return torch_sm(x)

    def full_pass():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full_pass, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = dict(
        kernel_name="softmax",
        x_name="N",
        x_label="hidden size",
        x_values=[128, 256, 512, 1024, 2048, 4096],
        kernel_providers=["liger", "torch"],
        extra_benchmark_configs=[
            {"M": 2048, "dtype": torch.float32},
            {"M": 2048, "dtype": torch.bfloat16},
        ],
    )

    run_benchmarks(
        bench_test_fn=bench_speed_softmax,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        overwrite=args.overwrite,
        **common_configs,
    )

    run_benchmarks(
        bench_test_fn=bench_memory_softmax,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        overwrite=args.overwrite,
        **common_configs,
    )