#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_layer_norm_speed.sh – end-to-end LayerNorm speed benchmark + SoL overlay
# -----------------------------------------------------------------------------
# 1. Runs benchmarks/scripts/benchmark_layer_norm.py for the forward path on
#    both the custom Triton kernel and PyTorch reference implementation.
# 2. Generates an annotated plot with a speed-of-light (memory-bound lower-bound)
#    line using the helper in scripts/plot_layer_norm_sol.py.
# 3. Stores artefacts under benchmarks/output/ for easy retrieval.
#
# Usage:
#   ./run_layer_norm_speed.sh                # uses defaults
#   BW=2900 ./run_layer_norm_speed.sh        # override sustained BW (GB/s)
#   MODES="forward backward" ./run_layer_norm_speed.sh  # run multiple modes
#
# Environment variables recognised:
#   BW     – sustained HBM bandwidth in GB/s (default 2850 for H100-PCIe)
#   MODES  – space-separated list of kernel modes to benchmark
#             forward | backward | full          (default: forward)
#   DEVICE – PyTorch device string (default: cuda)
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BW="${BW:-2850}"
MODES=( ${MODES:-forward} )
DEVICE="${DEVICE:-cuda}"

EXTRA_CFG='{"M": 4096, "dtype": "torch.float32", "eps": 1e-6}'
H_SIZES=(1024 2048 4096 8192 16384)
PROVIDERS=(custom torch)

DATA_DIR="${ROOT_DIR}/benchmarks/data"
VIS_DIR="${ROOT_DIR}/benchmarks/visualizations"
OUT_DIR="${ROOT_DIR}/benchmarks/output"
mkdir -p "${DATA_DIR}" "${VIS_DIR}" "${OUT_DIR}"

echo "[INFO] Running LayerNorm speed benchmarks (device=${DEVICE}, BW=${BW} GB/s)"
for MODE in "${MODES[@]}"; do
  if [[ "${MODE}" != "forward" ]]; then
    echo "[WARN] benchmark_layer_norm.py only supports forward; skipping mode '${MODE}'" >&2
    continue
  fi
  python "${ROOT_DIR}/benchmarks/scripts/benchmark_layer_norm.py" \
    --overwrite \
    >/dev/null

done

echo "[INFO] Generating plot(s) with speed-of-light overlay"
python <<'PY'
import json, os, sys, pathlib, math
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / 'benchmarks' / 'data' / 'all_benchmark_data.csv'
BW_GB_s = float(os.environ.get('BW', '2850'))
MODES = os.environ.get('MODES', 'forward').split()

BYTES_PER_ELT = {
    'forward': 16,
    'backward': 32,
    'full': 32,  # conservative upper-bound
}

df = pd.read_csv(DATA)
df["extra_benchmark_config"] = df["extra_benchmark_config_str"].apply(json.loads)

def sol_ms(mode, M):
    bytes_per = BYTES_PER_ELT[mode]
    return 1e3 * bytes_per * M / (BW_GB_s * 1e9)

for mode in MODES:
    sub = df[(df.kernel_name == 'layer_norm') & (df.kernel_operation_mode == mode) & (df.metric_name == 'speed')]
    if sub.empty:
        print(f"[WARN] No data for mode {mode}; skipping plot")
        continue

    # assume single extra config
    M = sub.extra_benchmark_config.iloc[0]['M']
    # Use pivot_table with mean aggregation to gracefully handle duplicate entries
    pivot = sub.pivot_table(index='x_value', columns='kernel_provider', values='y_value_50', aggfunc='mean')

    xs = pivot.index.tolist()
    sol_line = [sol_ms(mode, M)] * len(xs)

    plt.figure(figsize=(10,6))
    for provider in pivot.columns:
        plt.plot(xs, pivot[provider], marker='o', label=provider)
    plt.plot(xs, sol_line, linestyle='--', color='black', label='Speed-of-light')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Hidden size N')
    plt.ylabel('Latency (ms)')
    plt.title(f'LayerNorm {mode.capitalize()} – H100')
    plt.legend()
    plt.grid(True, which='both', ls=':')

    out_png = ROOT / 'benchmarks' / 'visualizations' / f'layer_norm_speed_{mode}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_png}")
PY

echo "[INFO] All done – results in ${VIS_DIR}" 