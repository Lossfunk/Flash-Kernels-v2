#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_diagonal_matmul_sol.sh – end-to-end Diagonal MatMul speed benchmark + SoL overlay
# -----------------------------------------------------------------------------
# 1. Runs benchmarks/scripts/benchmark_diagonal_matmul.py for the requested mode(s)
#    on both the custom CUDA kernel and the PyTorch reference implementation.
# 2. Generates an annotated plot with a speed-of-light (memory-bound lower-bound)
#    line using an inline Python helper.
# 3. Stores artefacts under benchmarks/output/ and benchmarks/visualizations/.
#
# Usage:
#   ./run_diagonal_matmul_sol.sh                     # uses defaults
#   BW=2900 ./run_diagonal_matmul_sol.sh             # override sustained BW (GB/s)
#   MODES="forward backward" ./run_diagonal_matmul_sol.sh  # run multiple modes
#
# Environment variables recognised:
#   BW     – Sustained HBM bandwidth in GB/s (default 2850 for H100-PCIe)
#   MODES  – Space-separated list of kernel modes to benchmark
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

# Extra config passed via benchmark script – mirrors defaults in benchmark_diagonal_matmul.py
EXTRA_CFG='{"M": 2048, "dtype": "torch.float32"}'
N_ROWS=(128 256 512 1024 2048 4096)
PROVIDERS=(custom torch)

DATA_DIR="${ROOT_DIR}/benchmarks/data"
VIS_DIR="${ROOT_DIR}/benchmarks/visualizations"
OUT_DIR="${ROOT_DIR}/benchmarks/output"
mkdir -p "${DATA_DIR}" "${VIS_DIR}" "${OUT_DIR}"

echo "[INFO] Running Diagonal MatMul speed benchmarks (device=${DEVICE}, BW=${BW} GB/s)"
for MODE in "${MODES[@]}"; do
  python "${ROOT_DIR}/benchmarks/scripts/benchmark_diagonal_matmul.py" \
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

# Approximate memory access bytes per element (output element) per mode
BYTES_PER_ELT = {
    'forward': 8,   # Read B + Write C (4B each)
    'backward': 12, # Read grad_out + Write grad_B (8B) + Read A (approx negligible per elt) + other accesses
    'full': 12,     # forward + backward dominated by backward cost
}

df = pd.read_csv(DATA)
# decode JSON column for ease of access
if 'extra_benchmark_config_str' in df.columns:
    df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

for mode in MODES:
    sub = df[(df.kernel_name == 'diag_matmul') & (df.kernel_operation_mode == mode) & (df.metric_name == 'speed')]
    if sub.empty:
        print(f"[WARN] No data for mode {mode}; skipping plot")
        continue

    # assume constant M across runs
    M = sub.extra_benchmark_config.iloc[0]['M'] if 'extra_benchmark_config' in sub.columns else 2048

    # Use pivot_table with mean aggregation
    pivot = sub.pivot_table(index='x_value', columns='kernel_provider', values='y_value_50', aggfunc='mean')

    xs = pivot.index.tolist()  # these are N values
    bytes_per = BYTES_PER_ELT[mode]
    sol_line = [1e3 * bytes_per * N * M / (BW_GB_s * 1e9) for N in xs]  # ms

    plt.figure(figsize=(10,6))
    for provider in pivot.columns:
        plt.plot(xs, pivot[provider], marker='o', label=provider)
    plt.plot(xs, sol_line, linestyle='--', color='black', label='Speed-of-light')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Rows N')
    plt.ylabel('Latency (ms)')
    plt.title(f'Diagonal MatMul {mode.capitalize()} – H100')
    plt.legend()
    plt.grid(True, which='both', ls=':')

    out_png = ROOT / 'benchmarks' / 'visualizations' / f'diag_matmul_speed_{mode}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_png}")
PY

echo "[INFO] All done – results in ${VIS_DIR}" 