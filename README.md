# Flash-Kernels v2 – Quick Reference

A lean playground for experimenting with **CUDA** GPU kernels.

Directory highlights
1. `benchmarks/` – micro-benchmarks + visualisations
2. `new_kernels/` – hand-tuned CUDA kernels (LayerNorm, Softmax, …)
3. `evals/` – PyTest correctness & regression suite
4. `agents/` – LLM pipeline that auto-writes kernels for KernelBench

---
## 1 · Installation
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt      # Dependencies including Triton (for benchmarking), PyTorch, plotting libs
```
GPU prerequisites
* CUDA 11.8+ drivers
* Compute Capability ≥ 7.0 (RTX 30-series, A100/H100, …)

---
## 2 · Running the tests (`evals/`)
```bash
pytest -q evals       # FP32 + BF16 when supported
```
Tests compare the CUDA kernels against the PyTorch reference with strict
`assert_verbose_allclose` tolerances.

---
## 3 · Benchmarking (`benchmarks/`)
### 3.1  One-liner drivers  🚀
For the common cases you do **not** need to pass any arguments – just execute
one of the convenience shell scripts and grab a coffee:
```bash
# Forward-only roofline runs
bash benchmarks/run_layer_norm_sol.sh
bash benchmarks/run_softmax_sol.sh
bash benchmarks/run_diagonal_matmul_sol.sh
bash benchmarks/run_fused_linear_rowsum_sol.sh
```
Each script will
1. call the corresponding `benchmarks/scripts/benchmark_*.py` file,
2. append results to `benchmarks/data/all_benchmark_data.csv`,
3. auto-generate PNGs in `benchmarks/visualizations/` (one per extra-config).

### 3.2  Manual control
Prefer explicit flags?  You can run the python scripts directly:
```bash
python benchmarks/scripts/benchmark_layer_norm.py --overwrite
python benchmarks/benchmark_visualizer.py \
       --kernel-name layer_norm --metric-name speed \
       --kernel-operation-mode forward --display
```
⚠️  *The visualiser iterates over every unique `extra_benchmark_config` unless
`--extra-config-filter` is supplied.*  Expect several PNGs.

---
## 4 · Reproducing the paper numbers
Full-scale regeneration (~25 min on an H100):
```bash
source venv/bin/activate  # ensure deps are present
for s in benchmarks/scripts/benchmark_*.py; do python "$s" --overwrite; done
for k in fused_linear_rowsum layer_norm softmax diag_matmul; do
  python benchmarks/benchmark_visualizer.py --kernel-name "$k" --metric-name speed
  python benchmarks/benchmark_visualizer.py --kernel-name "$k" --metric-name memory || true
done
```
Hardware: Ampere or newer GPU with ≥40 GB VRAM to cover the largest cases.
Generated `all_benchmark_data.csv` should match the committed copy (timestamps differ).
