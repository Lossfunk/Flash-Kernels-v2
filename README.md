# Flash-Kernels v2 – Quick Reference

A lean kernel-playground centred around three directories:

1. **`benchmarks/`** – Micro-benchmarks + visualisations for Triton kernels
2. **`evals/`**      – PyTest correctness & regression suite
3. **`new_kernels/`** – Our custom Triton kernels (LayerNorm, Softmax, …)

---

## 1 · Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # Triton ≥2,<3 + plotting deps
```
GPU prerequisites
* CUDA 11.8+ drivers
* Compute Capability ≥ 7.0 (RTX 30-series, A100/H100, …)

---

## 2 · Running the tests (`evals/`)
Execute the full test-suite (FP32 + BF16 where available):
```bash
pytest -q evals
```
Target files:
* `evals/test_layer_norm.py`
* `evals/test_softmax.py`

These use the functional wrappers under `new_kernels/**/Functional/` and compare against the PyTorch reference implementation with `assert_verbose_allclose`.

---

## 3 · Benchmarking (`benchmarks/`)
Record timing & memory results ⇒ CSV ⇒ PNG.

1. **Run a benchmark script** (overwrites the previous run by default):
   ```bash
   # LayerNorm speed – forward pass
   python benchmarks/scripts/benchmark_layer_norm.py --overwrite

   # Softmax (forward, backward, full)
   python benchmarks/scripts/benchmark_softmax.py --overwrite
   ```
   This appends rows to `benchmarks/data/all_benchmark_data.csv`.

2. **Generate a plot**
   ```bash
   python benchmarks/benchmark_visualizer.py \
     --kernel-name layer_norm \
     --metric-name speed \
     --kernel-operation-mode forward \
     --display      # optional GUI pop-up
   ```
   ➜ PNGs drop into `benchmarks/visualizations/`.

3. **One-shot helpers**
   There are ready-made driver scripts that benchmark _and_ plot with a roofline/SoL overlay:
   * `benchmarks/run_layer_norm_speed.sh`
   * `benchmarks/run_softmax_sol.sh`

---

## 4 · Kernel source (`new_kernels/`)
Directory layout (example LayerNorm):
```
new_kernels/
└─ layer_norm/
   ├─ Functional/               # Thin nn.Module wrapper (used by tests)
   │  └─ layer_norm.py
   └─ layer_norm.py             # Autograd-aware Triton kernel implementation
```
• **Functional** modules expose a PyTorch-friendly API.<br/>
• The root file contains the raw Triton kernel(s) plus backward hooks.

To add a new op:
1. Copy an existing folder (e.g. `layer_norm`) → rename.
2. Implement `*_forward` / `*_backward` functions in Triton.
3. Provide a `Functional/` wrapper and add tests under `evals/`.
4. Wire up a benchmark script under `benchmarks/scripts/`.

---

Happy hacking 🚀 
