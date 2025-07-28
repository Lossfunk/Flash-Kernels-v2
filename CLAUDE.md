# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flash-Kernels-v2 is a lean playground for experimenting with CUDA GPU kernels. It provides hand-optimized GPU kernels with performance benchmarking and correctness testing for common deep learning operations.

## Key Commands

### Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests (FP32 + BF16)
pytest -q evals

# Run specific kernel test
pytest evals/test_layer_norm.py
```

### Benchmarking
```bash
# Quick benchmark for a specific kernel
bash benchmarks/run_layer_norm_sol.sh

# Manual benchmark with visualization
python benchmarks/scripts/benchmark_layer_norm.py --overwrite
python benchmarks/benchmark_visualizer.py --kernel-name layer_norm --metric-name speed --display
```

## Architecture

### Core Components

1. **Kernel Implementations** (`new_kernels/`)
   - Each kernel has its own subdirectory: `layer_norm/`, `softmax/`, `diagonal_matmul/`, `fused_linear_rowsum/`
   - Each contains main implementation and `Functional/` subdirectory with functional interface
   - Kernels are written in CUDA C++ and compiled via PyTorch's cpp_extension.load_inline

2. **Testing** (`evals/`)
   - PyTest-based correctness tests comparing CUDA implementations against PyTorch references
   - Uses `utils.assert_verbose_allclose()` for strict tolerance checking
   - Tests run on both FP32 and BF16 (when hardware supports it)

3. **Benchmarking** (`benchmarks/`)
   - `scripts/` contains benchmark scripts for each kernel
   - Results stored in `data/` as CSV files
   - Visualizations generated in `visualizations/`
   - Supports custom (CUDA) and torch (PyTorch) providers
   - Uses Triton's benchmarking utilities (triton.testing.do_bench) for timing

### Key Implementation Patterns

- Kernels defined as CUDA C++ source strings (`_CUDA_SRC`)
- Compiled on-the-fly using `torch.utils.cpp_extension.load_inline`
- Forward and backward passes implemented separately
- Functional interfaces wrap the low-level kernels for easier use
- Device detection handles CUDA/XPU automatically via `utils.device_utils`

### Development Notes

- GPU Requirements: CUDA 11.8+, Compute Capability ≥ 7.0
- BF16 requires Ampere or newer (CC ≥ 8.0)
- Full benchmark suite needs ≥40 GB VRAM
- No build system - kernels compile on-the-fly via PyTorch's JIT compilation
- Speed-of-light overlays show theoretical memory bandwidth limits on performance graphs