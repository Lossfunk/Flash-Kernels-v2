from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
import time
import datetime
import random
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

import torch
import yaml  # For YAML persistence
from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.contracts import PerformanceIn, PerformanceOut, TensorSpec
from utils.logging_utils import get_logger

# Import from the new local proton_analyzer.py
from .proton_analyzer import ProtonRooflineAnalyzer, KernelMetrics, DeviceSpecs, MemoryAccessOptimizer

logger = get_logger("PerformanceAgent")
PERFORMANCE_OUTPUT_DIR = Path("performance_output")
PERFORMANCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Parent for individual op_hash subdirs
PROTON_TEMP_PROFILE_DIR_PARENT = Path("tmp_proton_profiles")
PROTON_TEMP_PROFILE_DIR_PARENT.mkdir(parents=True, exist_ok=True)

def _string_to_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int8": torch.int8,
        "bool": torch.bool,
    }
    return mapping.get(dtype_str.lower())

def _generate_test_inputs(input_specs: List[TensorSpec], device: str = "cuda") -> List[torch.Tensor]:
    logger.debug(f"Generating test inputs for {len(input_specs)} specs on device {device}")
    test_inputs = []
    for i, spec in enumerate(input_specs):
        logger.debug(f"Input {i}: shape={spec.shape}, dtype={spec.dtype}")
        if len(spec.shape) == 0:  # Scalar
            val = 1.0 if "float" in spec.dtype else 1
            if spec.dtype == "float16": tensor = torch.tensor(val, dtype=torch.float16, device=device)
            elif spec.dtype == "float32": tensor = torch.tensor(val, dtype=torch.float32, device=device)
            elif spec.dtype == "int32": tensor = torch.tensor(val, dtype=torch.int32, device=device)
            elif spec.dtype == "int8": tensor = torch.tensor(val, dtype=torch.int8, device=device)
            # Default scalar
            else: tensor = torch.tensor(val, dtype=torch.float32, device=device)
        else:  # Tensor
            shape_tuple = tuple(spec.shape)
            if spec.dtype == "float16": tensor = torch.randn(shape_tuple, dtype=torch.float16, device=device)
            elif spec.dtype == "float32": tensor = torch.randn(shape_tuple, dtype=torch.float32, device=device)
            # Adjusted range
            elif spec.dtype == "int32": tensor = torch.randint(0, 100, shape_tuple, dtype=torch.int32, device=device)
            elif spec.dtype == "int8": tensor = torch.randint(-128, 127, shape_tuple, dtype=torch.int8, device=device)
            # Default tensor
            else: tensor = torch.randn(shape_tuple, dtype=torch.float32, device=device)
        test_inputs.append(tensor)
    logger.info(f"Generated {len(test_inputs)} test inputs successfully.")
    return test_inputs


def _measure_pytorch_eager_baseline(pytorch_src: str, test_inputs: List[torch.Tensor], runs: int) -> float:
    logger.info(f"Measuring PyTorch eager baseline with {runs} runs.")

    # Indent the payload.pytorch_src to be a method of the class
    forward_method_lines = pytorch_src.strip().split('\n')
    indented_forward_method = []
    for i, line_content in enumerate(forward_method_lines):
        if line_content.strip():
            if i == 0: 
                indented_forward_method.append('    ' + line_content.lstrip())
            else:
                original_indent = len(line_content) - len(line_content.lstrip())
                indented_forward_method.append('    ' + ' ' * original_indent + line_content.lstrip())
        else: 
            indented_forward_method.append('')
    forward_method_str = "\n".join(indented_forward_method)

    reference_model_src = f"""
import torch
import torch.nn as nn
class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
{forward_method_str}
"""
    logger.debug(f"Generated PyTorch reference model source for baseline:\n{reference_model_src}")
    
    local_scope: Dict[str, Any] = {}
    try:
        exec(reference_model_src, {"torch": torch, "nn": torch.nn}, local_scope)
        ReferenceModelClass = local_scope['ReferenceModel']
        ref_model_instance = ReferenceModelClass().to(test_inputs[0].device if test_inputs else "cuda")
        
        # Warmup
        for _ in range(max(1, runs // 10)): # At least 1 warmup run
            with torch.no_grad(): 
                ref_model_instance.forward(*test_inputs)
        torch.cuda.synchronize()
        
        # Measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(runs):
            with torch.no_grad(): 
                ref_model_instance.forward(*test_inputs)
        end_event.record()
        torch.cuda.synchronize()
        
        baseline_time_ms = start_event.elapsed_time(end_event) / runs
        logger.info(f"PyTorch eager baseline: {baseline_time_ms:.3f} ms per run.")
        return baseline_time_ms
    except Exception as e:
        logger.error(f"Failed to execute PyTorch reference for baseline: {e}", exc_info=True)
        return -1.0 # Indicate error


async def _profile_and_analyze_kernel(payload: PerformanceIn) -> PerformanceOut:
    logger.info(f"PerformanceAgent profiling started for op_hash: {payload.op_hash}")
    op_hash_short = payload.op_hash[:12] # For shorter log/file names

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Performance profiling requires CUDA.")
        return _create_error_perf_out(payload.op_hash, "CUDA not available.")

    try:
        analyzer = ProtonRooflineAnalyzer(device="cuda")
        device_info = analyzer.get_device_info()
        logger.info(f"Device: {device_info.get('device_name', 'Unknown CUDA Device')}")
    except Exception as e:
        logger.error(f"Failed to initialize ProtonRooflineAnalyzer: {e}", exc_info=True)
        return _create_error_perf_out(payload.op_hash, f"ProtonRooflineAnalyzer init failed: {e}")

    # ---------------------------------------------------------------------
    # Reproducibility: set deterministic seeds
    # ---------------------------------------------------------------------
    try:
        seed_val = _deterministic_seed_from_op_hash(payload.op_hash)
        env_seed = os.getenv("PERF_AGENT_SEED")
        if env_seed is not None:
            seed_val = int(env_seed)
        _set_global_seeds(seed_val)
    except Exception as seed_exc:
        logger.warning(f"Failed to set deterministic seed: {seed_exc}")

    # Instantiate MemoryAccessOptimizer
    memory_optimizer = MemoryAccessOptimizer(analyzer.device_specs)
    all_memory_access_analysis = {}
    try:
        for i, spec in enumerate(payload.input_specs):
            torch_dtype = _string_to_torch_dtype(spec.dtype)
            if torch_dtype:
                analysis = memory_optimizer.analyze_access_pattern(tuple(spec.shape), torch_dtype)
                all_memory_access_analysis[f"input_{i}_{spec.dtype}_{'_'.join(map(str,spec.shape))}"] = analysis
            else:
                logger.warning(f"Could not convert dtype string '{spec.dtype}' to torch.dtype for memory analysis of input {i}.")
    except Exception as e:
        logger.error(f"Error during memory access analysis: {e}", exc_info=True)
        # Continue, but memory_access_analysis might be incomplete or empty

    # 1. Load Triton kernel dynamically
    triton_module = None
    launcher_func: Optional[Callable] = None
    kernel_name_for_profile = f"triton_kernel_{op_hash_short}"
    try:
        if not Path(payload.source_file_path).exists():
            return _create_error_perf_out(payload.op_hash, f"Triton source file not found: {payload.source_file_path}")

        spec = importlib.util.spec_from_file_location(f"triton_user_kernel_{op_hash_short}", payload.source_file_path)
        if spec is None or spec.loader is None:
            return _create_error_perf_out(payload.op_hash, f"Could not get import spec for {payload.source_file_path}")
        
        triton_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_module)
        
        triton_kernel_func = None
        for name in dir(triton_module):
            obj = getattr(triton_module, name)
            if name.startswith("launch_") and callable(obj):
                # Found a launcher function
                launcher_func = obj
                kernel_name_for_profile = name
                logger.info(f"Found Triton launcher function: {kernel_name_for_profile}")
                break
            elif hasattr(obj, '_original_fn'): # Check for JITFunction (actual Triton kernel)
                triton_kernel_func = obj
                logger.info(f"Found Triton kernel function: {name}")
        
        # If no launch_ function found, look for any function that calls a kernel
        if launcher_func is None:
            import ast
            try:
                # Read the source file to parse it
                with open(payload.source_file_path, 'r') as f:
                    source_content = f.read()
                tree = ast.parse(source_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if this function contains kernel calls (kernel[grid](...))
                        for child in ast.walk(node):
                            if isinstance(child, ast.Subscript):
                                # Look for pattern like kernel_name[grid](...)
                                if isinstance(child.value, ast.Name):
                                    # This function calls a kernel, use it as launcher
                                    if hasattr(triton_module, node.name):
                                        launcher_func = getattr(triton_module, node.name)
                                        kernel_name_for_profile = node.name
                                        logger.info(f"Found kernel-calling function: {kernel_name_for_profile}")
                                        break
                        if launcher_func:
                            break
            except Exception as e:
                logger.warning(f"Failed to parse kernel source for function detection: {e}")
        
        if launcher_func is None and triton_kernel_func is None:
            return _create_error_perf_out(payload.op_hash, "No launcher function (launch_...) or @triton.jit kernel found in Triton source.")
    except Exception as e:
        logger.error(f"Failed to load Triton kernel from {payload.source_file_path}: {e}", exc_info=True)
        return _create_error_perf_out(payload.op_hash, f"Triton kernel load failed: {e}")

    # 2. Generate test inputs
    try:
        test_inputs = _generate_test_inputs(payload.input_specs, device="cuda")
        if not test_inputs: return _create_error_perf_out(payload.op_hash, "Failed to generate any test inputs.")

        ref_tensor_for_output = test_inputs[0]
        output_tensor = torch.empty_like(ref_tensor_for_output)
        kernel_args_for_profiling: Tuple = (output_tensor, *test_inputs)

    except Exception as e:
        logger.error(f"Failed to generate test inputs or prepare kernel args: {e}", exc_info=True)
        return _create_error_perf_out(payload.op_hash, f"Test input generation failed: {e}")

    # 3. Measure PyTorch eager baseline
    baseline_eager_time_ms = _measure_pytorch_eager_baseline(payload.pytorch_src, test_inputs, payload.runs)
    if baseline_eager_time_ms < 0:
        logger.warning("PyTorch eager baseline measurement failed. Speedup will be 0.")
        baseline_eager_time_ms = 0.0 # Avoid negative speedup

    # 4. Profile Triton kernel
    triton_kernel_metrics: Optional[KernelMetrics] = None # Use KernelMetrics from proton_analyzer
    recommendations: List[str] = ["Profiling did not complete or was skipped."]
    
    current_op_profile_dir = PROTON_TEMP_PROFILE_DIR_PARENT / payload.op_hash
    current_op_profile_dir.mkdir(parents=True, exist_ok=True)

    # Let Proton measure FLOPS/bytes automatically from hardware counters
    # No need for hardcoded estimates - Proton will extract actual values
    logger.info("Using Proton to automatically measure FLOPS and bytes from hardware counters")

    try:
        logger.info(f"Profiling Triton kernel '{kernel_name_for_profile}' with {payload.runs} iterations.")
        
        # -----------------------------------------------------------------
        # Kernel launching logic - handle launcher function vs direct kernel
        # -----------------------------------------------------------------
        def triton_kernel_launcher_for_profiling():
            if launcher_func is not None:
                # Check if it's a traditional launcher function or regular function
                if hasattr(launcher_func, '__name__') and launcher_func.__name__.startswith('launch_'):
                    # Traditional launcher function - doesn't return anything, modifies output in-place
                    launcher_func(*kernel_args_for_profiling)
                    return None  # No return value expected
                else:
                    # Regular function that returns the result
                    return launcher_func(*test_inputs)  # Use test_inputs, not kernel_args_for_profiling
            elif triton_kernel_func is not None:
                # Use the Triton kernel directly with grid
                try:
                    grid_arg = _resolve_triton_grid(triton_kernel_func, test_inputs[0], kernel_args_for_profiling)
                    logger.info(f"Using Triton grid: {grid_arg}")
                except Exception as e_grid:
                    logger.warning(f"_resolve_triton_grid failed: {e_grid}. Falling back to (1,)")
                    grid_arg = (1,)
                return triton_kernel_func[grid_arg](*kernel_args_for_profiling)
            else:
                raise RuntimeError("No valid kernel or launcher function found")

        triton_kernel_metrics = analyzer.profile_kernel(
            kernel_func=triton_kernel_launcher_for_profiling,
            args=(), 
            kernel_name=kernel_name_for_profile,
            flops=0.0,  # Let Proton measure automatically
            bytes_transferred=0.0,  # Let Proton measure automatically
            warmup_iters=max(1, payload.runs // 10),
            profile_iters=payload.runs,
            output_dir=str(current_op_profile_dir) 
        )
        
        if triton_kernel_metrics:
            logger.info(f"Triton kernel profiled. Runtime: {triton_kernel_metrics.runtime_ms:.3f} ms")
            # generate_optimization_report now returns a string report, not just recommendations list
            # We might want to adjust this if only a list of strings is needed for PerformanceOut.recommendations
            # For now, let's get the full report and then decide if we split it or pass it differently.
            # The generate_optimization_report in proton_analyzer.py saves a YAML and returns a string.
            # The string report is what we need for recommendations.
            full_report_str = analyzer.generate_optimization_report(triton_kernel_metrics, output_dir=str(current_op_profile_dir))
            # Extract recommendations from the string report if needed, or adjust PerformanceOut
            # For now, splitting the report string into lines for recommendations for simplicity
            recommendations = full_report_str.split('\n')
        else:
            logger.warning("Triton profiling returned no metrics.")
            recommendations = ["Triton profiling returned no metrics."]
            triton_kernel_metrics = KernelMetrics(name=kernel_name_for_profile, runtime_ms=-1.0, flops=0, bytes_transferred=0, arithmetic_intensity=0, achieved_bandwidth_gb_s=0, achieved_flops_gflops=0, occupancy_percent=0, bottleneck="error", efficiency_percent=0, roofline_position=(0,0), raw_proton_metrics=None, profile_hatchet_path=None, yaml_report_path=None)

    except Exception as e:
        logger.error(f"Error during Triton kernel profiling for {payload.op_hash}: {e}", exc_info=True)
        recommendations = [f"Triton profiling failed: {str(e)}"]
        triton_kernel_metrics = KernelMetrics(name=kernel_name_for_profile, runtime_ms=-1.0, flops=0, bytes_transferred=0, arithmetic_intensity=0, achieved_bandwidth_gb_s=0, achieved_flops_gflops=0, occupancy_percent=0, bottleneck="error", efficiency_percent=0, roofline_position=(0,0), raw_proton_metrics=None, profile_hatchet_path=None, yaml_report_path=None)

    triton_runtime_ms = triton_kernel_metrics.runtime_ms if triton_kernel_metrics and triton_kernel_metrics.runtime_ms > 0 else 0.0
    speedup = 0.0
    if baseline_eager_time_ms > 0 and triton_runtime_ms > 0:
        speedup = baseline_eager_time_ms / triton_runtime_ms
    logger.info(f"Speedup: {speedup:.2f}x (Baseline: {baseline_eager_time_ms:.3f}ms, Triton: {triton_runtime_ms:.3f}ms)")

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Use the yaml_report_path from KernelMetrics if available, otherwise generate one.
    final_yaml_path = triton_kernel_metrics.yaml_report_path if triton_kernel_metrics and triton_kernel_metrics.yaml_report_path else str(PERFORMANCE_OUTPUT_DIR / f"{payload.op_hash}_{timestamp_str}_fallback.yaml")
    
    # If the report wasn't saved by generate_optimization_report (e.g. if output_dir wasn't passed or error),
    # we might need to save it here or ensure generate_optimization_report *always* saves if it can.
    # The current proton_analyzer.generate_optimization_report saves if output_dir is given.
    # We passed current_op_profile_dir to it, so it should have saved there.
    # We need to ensure final_yaml_path points to where generate_optimization_report saved it.
    # The generate_optimization_report saves it as {metrics.name}_perf_report.yaml in the output_dir given to it.
    # So, the path should be current_op_profile_dir / f"{triton_kernel_metrics.name}_perf_report.yaml"
    if triton_kernel_metrics and triton_kernel_metrics.name:
        expected_yaml_path_from_analyzer = current_op_profile_dir / f"{triton_kernel_metrics.name}_perf_report.yaml"
        if expected_yaml_path_from_analyzer.exists():
            final_yaml_path = str(expected_yaml_path_from_analyzer.resolve())
        elif triton_kernel_metrics.yaml_report_path: # Fallback to the one stored in metrics if somehow different
             final_yaml_path = triton_kernel_metrics.yaml_report_path
        # If still no valid path, we might need to save a summary YAML here.
        # For now, we assume generate_optimization_report handled it, or final_yaml_path remains the fallback.

    # If generate_optimization_report did not run or save, we still might want to save basic yaml_data
    # The current structure of generate_optimization_report saves it, so this might be redundant unless error cases.
    # Let's ensure yaml_data is prepared for PerformanceOut even if not re-saving here.
    yaml_data_for_perf_out = {
        "op_hash": payload.op_hash,
        "timestamp": datetime.datetime.now().isoformat(),
        "source_file_path": payload.source_file_path,
        "pytorch_src_hash": hashlib.md5(payload.pytorch_src.encode()).hexdigest(),
        "input_specs": [spec.model_dump() for spec in payload.input_specs],
        "profiling_runs": payload.runs,
        "estimated_flops": triton_kernel_metrics.flops if triton_kernel_metrics else 0.0,
        "estimated_bytes_transferred": triton_kernel_metrics.bytes_transferred if triton_kernel_metrics else 0.0,
        "baseline_eager_time_ms": baseline_eager_time_ms,
        "device_specs": device_info if 'device_info' in locals() and device_info else analyzer.get_device_info(),
        "kernel_metrics": triton_kernel_metrics.model_dump() if triton_kernel_metrics and hasattr(triton_kernel_metrics, 'model_dump') else {},
        "speedup": speedup,
        "recommendations": recommendations,
        "proton_profile_hatchet_path": triton_kernel_metrics.profile_hatchet_path if triton_kernel_metrics else None,
        "raw_proton_metrics": triton_kernel_metrics.raw_proton_metrics if triton_kernel_metrics else None,
        "memory_access_analysis": all_memory_access_analysis,
        "final_yaml_report_path_for_agent": final_yaml_path
    }
    # Save a consolidated YAML by PerformanceAgent if generate_optimization_report didn't or for consistency.
    # This YAML will go into PERFORMANCE_OUTPUT_DIR, not the op_hash specific subdir.
    agent_yaml_filename = f"{payload.op_hash}_{timestamp_str}_agent_summary.yaml"
    agent_yaml_path = PERFORMANCE_OUTPUT_DIR / agent_yaml_filename
    try:
        with open(agent_yaml_path, 'w') as f:
            yaml.dump(yaml_data_for_perf_out, f, indent=2, sort_keys=False)
        logger.info(f"PerformanceAgent summary report saved to: {agent_yaml_path}")
        final_yaml_path = str(agent_yaml_path.resolve()) # PerformanceOut should point to this summary
    except Exception as e:
        logger.error(f"Failed to save PerformanceAgent summary YAML to {agent_yaml_path}: {e}", exc_info=True)
        # final_yaml_path will retain its previous value (either from proton_analyzer or fallback)

    return PerformanceOut(
        runtime_ms=triton_kernel_metrics.runtime_ms if triton_kernel_metrics else -1.0,
        speedup=speedup,
        achieved_gflops=triton_kernel_metrics.achieved_flops_gflops if triton_kernel_metrics else 0.0,
        achieved_bandwidth_gb_s=triton_kernel_metrics.achieved_bandwidth_gb_s if triton_kernel_metrics else 0.0,
        efficiency_percent=triton_kernel_metrics.efficiency_percent if triton_kernel_metrics else 0.0,
        occupancy_percent=triton_kernel_metrics.occupancy_percent if triton_kernel_metrics else 0.0, 
        recommendations=recommendations,
        yaml_path=final_yaml_path, 
        device_specs=yaml_data_for_perf_out["device_specs"], 
        kernel_metrics_raw=yaml_data_for_perf_out["kernel_metrics"], # This is the full KernelMetrics model_dump
        raw_proton_metrics=triton_kernel_metrics.raw_proton_metrics if triton_kernel_metrics else None,
        profile_hatchet_path=triton_kernel_metrics.profile_hatchet_path if triton_kernel_metrics else None,
        memory_access_analysis=all_memory_access_analysis
    )

def _create_error_perf_out(op_hash: str, error_message: str) -> PerformanceOut:
    logger.error(f"Creating error PerformanceOut for {op_hash}: {error_message}")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    yaml_filename = f"{op_hash}_{timestamp_str}_error.yaml"
    yaml_path = PERFORMANCE_OUTPUT_DIR / yaml_filename
    error_data = {
        "op_hash": op_hash,
        "timestamp": datetime.datetime.now().isoformat(),
        "error": error_message,
        "recommendations": [f"Profiling critical error: {error_message}"]
    }
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(error_data, f, indent=2)
        logger.info(f"Error report YAML saved to: {yaml_path}")
        final_yaml_path = str(yaml_path.resolve())
    except Exception as e_yaml:
        logger.error(f"Failed to write error YAML for {op_hash}: {e_yaml}")
        final_yaml_path = ""

    return PerformanceOut(
        runtime_ms=-1.0,
        speedup=0.0,
        achieved_gflops=0.0,
        achieved_bandwidth_gb_s=0.0,
        efficiency_percent=0.0,
        occupancy_percent=0.0,
        recommendations=[f"Profiling failed: {error_message}"],
        yaml_path=final_yaml_path, 
        device_specs={"error": error_message},
        kernel_metrics_raw={"error": error_message},
        raw_proton_metrics=None, # Initialize new field
        profile_hatchet_path=None, # Initialize new field
        memory_access_analysis=None # Initialize new field
    )

performance_tool = FunctionTool(_profile_and_analyze_kernel)

class PerformanceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="performance_analyzer",
            description="Profiles Triton kernels, compares against PyTorch eager execution, and provides performance optimization recommendations.",
            tools=[performance_tool]
        )

    async def profile(self, payload: PerformanceIn) -> PerformanceOut:
        logger.info(f"PerformanceAgent.profile invoked for op_hash: {payload.op_hash}")
        try:
            if not all([payload.op_hash, payload.ptx_path, payload.source_file_path, payload.input_specs, payload.pytorch_src]):
                 missing_fields = [field for field in ["op_hash", "ptx_path", "source_file_path", "input_specs", "pytorch_src"] if not getattr(payload, field)]
                 err_msg = f"Missing required fields in PerformanceIn: {missing_fields}"
                 logger.error(err_msg)
                 return _create_error_perf_out(payload.op_hash or "unknown_ophash", err_msg)
            
            return await _profile_and_analyze_kernel(payload)
        except Exception as e:
            logger.error(f"Unhandled error during performance analysis for op_hash {payload.op_hash or 'unknown_ophash'}: {e}", exc_info=True)
            return _create_error_perf_out(payload.op_hash or "unknown_ophash", f"Unhandled agent error: {e}") 

# === Determinism helpers =====================================================

def _deterministic_seed_from_op_hash(op_hash: str) -> int:
    """Derive a deterministic integer seed from the first 8 hex chars of the op_hash.
    Falls back to 42 if conversion fails."""
    try:
        return int(hashlib.sha256(op_hash.encode()).hexdigest()[:8], 16)
    except Exception:
        return 42


def _set_global_seeds(seed: int) -> None:
    """Seed Python, torch (CPU & CUDA) RNGs for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Disable nondeterministic CuDNN algorithms for extra safety
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    logger.info(f"Deterministic seed set to {seed}")

# === Grid resolution helpers ==================================================

def _resolve_triton_grid(launcher_func: Callable, first_input: torch.Tensor, example_args: Tuple[Any, ...]) -> Tuple[int, ...]:
    """Attempt to infer a sensible Triton launch grid.

    Resolution order:
      1. PERF_AGENT_GRID env-var override (comma-separated ints)
      2. `default_grid` attribute exposed by launcher function
      3. Triton meta-grid heuristic if available (`launcher_func.meta['grid']`)
      4. Fallback 1-D grid proportional to input.numel().
    """
    # 1) Environment override ==================================================
    env_grid = os.getenv("PERF_AGENT_GRID")
    if env_grid:
        try:
            grid_tuple = tuple(int(x) for x in env_grid.split(',') if x)
            if grid_tuple:
                return grid_tuple
        except ValueError as e:
            logger.warning(f"Invalid PERF_AGENT_GRID '{env_grid}': {e}. Ignoring override.")

    # 2) Explicit attribute =====================================================
    if hasattr(launcher_func, "default_grid"):
        attr_val = getattr(launcher_func, "default_grid")
        try:
            if callable(attr_val):
                grid_val = tuple(attr_val(*example_args))  # type: ignore[arg-type]
            else:
                grid_val = tuple(attr_val)
            if grid_val:
                return grid_val
        except Exception as e:
            logger.warning(f"default_grid resolution failed: {e}")

    # 3) Triton meta grid =======================================================
    if hasattr(launcher_func, "meta") and isinstance(getattr(launcher_func, "meta"), dict):
        meta_dict = getattr(launcher_func, "meta")
        grid_obj = meta_dict.get("grid")
        if callable(grid_obj):
            try:
                grid_val = tuple(grid_obj({}))  # meta function signature lenient
                if grid_val:
                    return grid_val
            except Exception as e:
                logger.debug(f"launcher_func.meta['grid'] call failed: {e}")

    # 4) Simple 1-D fallback ====================================================
    total_elements = int(first_input.numel()) if isinstance(first_input, torch.Tensor) else 0
    block_size = 256
    num_blocks = max(1, (total_elements + block_size - 1) // block_size)
    return (num_blocks,) 
