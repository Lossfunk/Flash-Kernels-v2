"""
python benchmarks_visualizer.py \
    --kernel-name kto_loss \
    --metric-name speed \
    --kernel-operation-mode forward backward
"""

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

from new_kernels.softmax.Functional.softmax import Softmax


device = infer_device()


# -----------------------------------------------------------------------------
# NOTE: The generic torch.utils.benchmark utilities have been removed in favour
# of Triton's built-in do_bench which directly returns the desired quantiles.
# -----------------------------------------------------------------------------


def bench_speed_softmax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark speed (ms) for Softmax forward/backward/full passes."""
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra = input.extra_benchmark_config or {}
    M = extra.get("M", 2048)
    dtype = extra.get("dtype", torch.float32)

    x_shape = (M, N)

    custom_softmax = Softmax().to(device).to(dtype)
    torch_softmax = torch.nn.Softmax(dim=-1).to(device).to(dtype)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "custom":
            return custom_softmax(x)
        if provider == "torch":
            return torch_softmax(x)

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

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )
    else:
        raise ValueError(f"Unknown kernel operation mode: {mode}")

    # Sanity check – Triton returns `None` on warning/error. Fail fast.
    if any(val is None for val in (ms_20, ms_50, ms_80)):
        raise RuntimeError(
            f"Benchmark speed result is None: ms_20={ms_20}, ms_50={ms_50}, ms_80={ms_80}"
        )

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_softmax(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark memory footprint (MB) of a full Softmax forward+backward pass."""
    N = input.x
    provider = input.kernel_provider
    extra = input.extra_benchmark_config or {}
    dtype = extra.get("dtype", torch.float32)
    M = extra.get("M", 2048)

    x_shape = (M, N)

    custom_softmax = Softmax().to(device).to(dtype)
    torch_softmax = torch.nn.Softmax(dim=-1).to(device).to(dtype)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "custom":
            return custom_softmax(x)
        if provider == "torch":
            return torch_softmax(x)

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
        kernel_providers=["custom", "torch"],
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