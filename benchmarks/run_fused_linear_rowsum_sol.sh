#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_fused_linear_rowsum_sol.sh – end-to-end FusedLinearRowSum speed benchmark + SoL overlay
# -----------------------------------------------------------------------------
# 1. Runs benchmarks/scripts/benchmark_fused_linear_rowsum.py for the forward path on
#    both the custom fused kernel and PyTorch reference implementation.
# 2. Generates an annotated plot with a speed-of-light (memory-bound lower-bound)
#    line using the helper in scripts/plot_fused_linear_rowsum_sol.py.
# 3. Stores artefacts under benchmarks/output/ for easy retrieval.
#
# Usage:
#   ./run_fused_linear_rowsum_sol.sh                     # uses defaults
#   BW=2900 ./run_fused_linear_rowsum_sol.sh             # override sustained BW (GB/s)
#   MODES="forward backward" ./run_fused_linear_rowsum_sol.sh  # run multiple modes
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

EXTRA_CFG='{"batch_size": 1024, "output_dim": 2048, "dtype": "torch.float32", "bias": true}'
INPUT_DIMS=(512 1024 2048 4096 8192)
PROVIDERS=(custom torch)

DATA_DIR="${ROOT_DIR}/benchmarks/data"
VIS_DIR="${ROOT_DIR}/benchmarks/visualizations"
OUT_DIR="${ROOT_DIR}/benchmarks/output"
mkdir -p "${DATA_DIR}" "${VIS_DIR}" "${OUT_DIR}"

echo "[INFO] Running FusedLinearRowSum speed benchmarks (device=${DEVICE}, BW=${BW} GB/s)"
for MODE in "${MODES[@]}"; do
  python "${ROOT_DIR}/benchmarks/scripts/benchmark_fused_linear_rowsum.py" \
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

# FusedLinearRowSum memory access patterns (bytes per element for different modes)
# For fused linear rowsum: input (batch_size, input_dim), weight (output_dim, input_dim), 
# bias (output_dim), output (batch_size) - but we reduce to scalar per batch
BYTES_PER_ELT = {
    'forward': 12,   # Read input (4B) + Read weight (4B) + Read bias (4B) - optimized for rowsum
    'backward': 20,  # Read grad_output (4B) + Read input (4B) + Read weight (4B) + Write grad_input (4B) + Write grad_weight (4B) + Write grad_bias (4B)
    'full': 20,      # Combined forward + backward pass
}

df = pd.read_csv(DATA)
df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

def sol_ms(mode, batch_size, input_dim, output_dim):
    """Calculate speed-of-light latency for fused linear rowsum operations."""
    bytes_per = BYTES_PER_ELT[mode]
    # Total memory access: input + weight + bias (forward), plus gradients (backward)
    if mode == 'forward':
        total_bytes = (batch_size * input_dim + output_dim * input_dim + output_dim) * 4  # 4 bytes per float32
    else:  # backward or full
        total_bytes = (batch_size * input_dim * 2 + output_dim * input_dim * 2 + output_dim * 2) * 4
    
    return 1e3 * total_bytes / (BW_GB_s * 1e9)

for mode in MODES:
    sub = df[(df.kernel_name == 'fused_linear_rowsum') & (df.kernel_operation_mode == mode) & (df.metric_name == 'speed')]
    if sub.empty:
        print(f"[WARN] No data for mode {mode}; skipping plot")
        continue

    # Get configuration from the first entry
    config = sub.extra_benchmark_config.iloc[0]
    batch_size = config['batch_size']
    output_dim = config['output_dim']
    
    # Use pivot_table with mean aggregation to gracefully handle duplicate entries
    pivot = sub.pivot_table(index='x_value', columns='kernel_provider', values='y_value_50', aggfunc='mean')

    xs = pivot.index.tolist()
    sol_line = [sol_ms(mode, batch_size, input_dim, output_dim) for input_dim in xs]

    plt.figure(figsize=(10,6))
    for provider in pivot.columns:
        plt.plot(xs, pivot[provider], marker='o', label=provider)
    plt.plot(xs, sol_line, linestyle='--', color='black', label='Speed-of-light')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Input Dimension')
    plt.ylabel('Latency (ms)')
    plt.title(f'FusedLinearRowSum {mode.capitalize()} – H100')
    plt.legend()
    plt.grid(True, which='both', ls=':')

    out_png = ROOT / 'benchmarks' / 'visualizations' / f'fused_linear_rowsum_speed_{mode}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_png}")
PY

echo "[INFO] All done – results in ${VIS_DIR}" 