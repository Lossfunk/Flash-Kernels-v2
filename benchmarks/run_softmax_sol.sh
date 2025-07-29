#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_softmax_sol.sh – end-to-end Softmax speed benchmark + SoL overlay
# -----------------------------------------------------------------------------
# 1. Runs benchmarks/scripts/benchmark_softmax.py for the forward path on
#    both the custom CUDA kernel and PyTorch reference implementation.
# 2. Generates an annotated plot with a speed-of-light (memory-bound lower-bound)
#    line using the helper in scripts/plot_softmax_sol.py.
# 3. Stores artefacts under benchmarks/output/ for easy retrieval.
#
# Usage:
#   ./run_softmax_sol.sh                     # uses defaults
#   BW=2900 ./run_softmax_sol.sh             # override sustained BW (GB/s)
#   MODES="forward backward" ./run_softmax_sol.sh  # run multiple modes
#
# Environment variables recognised:
#   BW     – sustained HBM bandwidth in GB/s (default 2850 for H100-PCIe)
#   MODES  – space-separated list of kernel modes to benchmark
#             forward | backward | full          (default: forward)
#   DEVICE – PyTorch device string (default: cuda)
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT_DIR  # make ROOT_DIR available to child processes (inline Python)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BW="${BW:-2850}"
MODES=( ${MODES:-forward} )
DEVICE="${DEVICE:-cuda}"

EXTRA_CFG='{"M": 2048, "dtype": "torch.float32"}'
H_SIZES=(128 256 512 1024 2048 4096)
PROVIDERS=(custom torch)

DATA_DIR="${ROOT_DIR}/benchmarks/data"
VIS_DIR="${ROOT_DIR}/benchmarks/visualizations"
OUT_DIR="${ROOT_DIR}/benchmarks/output"
mkdir -p "${DATA_DIR}" "${VIS_DIR}" "${OUT_DIR}"

echo "[INFO] Running Softmax speed benchmarks (device=${DEVICE}, BW=${BW} GB/s)"
for MODE in "${MODES[@]}"; do
  python "${ROOT_DIR}/benchmarks/scripts/benchmark_softmax.py" \
    --overwrite \
    >/dev/null
done

echo "[INFO] Generating plot(s) with speed-of-light overlay"
python <<'PY'
import json, os, sys, pathlib, math
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(os.environ.get('ROOT_DIR', '.')).resolve()
DATA = ROOT / 'benchmarks' / 'data' / 'all_benchmark_data.csv'
BW_GB_s = float(os.environ.get('BW', '2850'))
MODES = os.environ.get('MODES', 'forward').split()

# Softmax memory access patterns (bytes per element for different modes)
# For softmax: input (M, N), output (M, N), intermediate max/sum values
BYTES_PER_ELT = {
    'forward': 8,    # Read input (4B) + Write output (4B) - optimized single pass
    'backward': 12,  # Read input (4B) + Read grad_output (4B) + Write grad_input (4B)
    'full': 12,      # Combined forward + backward pass
}

df = pd.read_csv(DATA)
df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

def sol_ms(mode, M, N):
    """Calculate speed-of-light latency for softmax operations."""
    bytes_per = BYTES_PER_ELT[mode]
    total_elements = M * N
    total_bytes = bytes_per * total_elements
    return 1e3 * total_bytes / (BW_GB_s * 1e9)

for mode in MODES:
    sub = df[(df.kernel_name == 'softmax') & (df.kernel_operation_mode == mode) & (df.metric_name == 'speed')]
    if sub.empty:
        print(f"[WARN] No data for mode {mode}; skipping plot")
        continue

    # Get M from the first configuration
    M = sub.extra_benchmark_config.iloc[0]['M']
    # Use pivot_table with mean aggregation to gracefully handle duplicate entries
    pivot = sub.pivot_table(index='x_value', columns='kernel_provider', values='y_value_50', aggfunc='mean')

    xs = pivot.index.tolist()
    sol_line = [sol_ms(mode, M, N) for N in xs]

    plt.figure(figsize=(10,6))
    for provider in pivot.columns:
        plt.plot(xs, pivot[provider], marker='o', label=provider)
    plt.plot(xs, sol_line, linestyle='--', color='black', label='Speed-of-light')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Hidden size N')
    plt.ylabel('Latency (ms)')
    plt.title(f'Softmax {mode.capitalize()} – H100')
    plt.legend()
    plt.grid(True, which='both', ls=':')

    out_png = ROOT / 'benchmarks' / 'visualizations' / f'softmax_speed_{mode}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_png}")
PY

echo "[INFO] All done – results in ${VIS_DIR}"
