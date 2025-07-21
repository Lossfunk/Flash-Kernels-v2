import torch
import triton
import sys
import os

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import QUANTILES
from utils.utils import SingleBenchmarkRunInput
from utils.utils import SingleBenchmarkRunOutput
from utils.utils import _test_memory
from utils.utils import parse_benchmark_script_args
from utils.utils import run_benchmarks
from utils.utils import infer_device

from new_kernels.layer_norm.Functional.layer_norm import LayerNorm

device = infer_device()

# -----------------------------------------------------------------------------
# Removed torch.utils.benchmark utilities. We now rely on Triton's do_bench
# which offers quantile statistics directly.
# -----------------------------------------------------------------------------

def bench_speed_layer_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)
    triton_ln = LayerNorm(hidden_size=N).to(device)
    torch_ln = torch.nn.LayerNorm(N, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "custom":
            return triton_ln(x)
        if provider == "torch":
            return torch_ln(x)
            
    # Warm-up run to deal with GPU initialization overhead
    for _ in range(5):
        y_fwd()
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
        # Warm-up for backward
        for _ in range(5):
            y_fwd().backward(dy, retain_graph=True)
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

        # Warm-up for full
        for _ in range(5):
            full()
        torch.cuda.synchronize()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=500,
        )

    # Fail fast if Triton produced any None metrics
    if any(val is None for val in (ms_20, ms_50, ms_80)):
        raise RuntimeError(
            f"Benchmark speed result is None: ms_20={ms_20}, ms_50={ms_50}, ms_80={ms_80}"
        )

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)

"""
def bench_memory_layer_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    dtype = input.extra_benchmark_config["dtype"]
    M = input.extra_benchmark_config["M"]
    eps = input.extra_benchmark_config["eps"]

    x_shape = (M, N)

    triton_ln = LayerNorm(hidden_size=N).to(device)
    torch_ln = torch.nn.LayerNorm(N, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    def y_fwd():
        if provider == "liger":
            return triton_ln(x)
        if provider == "huggingface":
            return torch_ln(x)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )
"""

if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "layer_norm",
        "x_name": "N",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 15)],
        "kernel_providers": ["custom", "torch"],
        "extra_benchmark_configs": [{"M": 4096, "dtype": torch.float32, "eps": 1e-6}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_layer_norm,
        kernel_operation_modes=["forward"], # Can use forward, backward, full
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    """
    run_benchmarks(
        bench_test_fn=bench_memory_layer_norm,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
    """