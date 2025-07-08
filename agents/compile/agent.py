from __future__ import annotations

import tempfile
import importlib.util
from pathlib import Path
import datetime
import random
import os
import triton  # ensure main module available
# ----- Triton compatibility imports -----
# Triton reorganised some internal paths between major versions (e.g. 2.x vs 3.x).
# We try the new path first and fall back gracefully so the agent keeps working
# irrespective of the exact minor version within the 2.x series.

try:
    # Triton 2.1+ provides JITFunction at this location
    from triton.runtime.jit import JITFunction  # type: ignore
except ImportError:  # pragma: no cover - older/newer layout
    from triton import JITFunction  # type: ignore

# Import Autotuner for @triton.autotune detection
try:
    from triton.runtime.autotuner import Autotuner  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from triton.autotuner import Autotuner  # type: ignore
    except ImportError:
        Autotuner = None  # type: ignore

# Target descriptor – name changed between versions.
try:
    # Triton 3.x (used previously)
    from triton.compiler.backends.cuda import CUDATarget  # type: ignore
except ImportError:  # pragma: no cover
    try:
        # Triton 2.x - use CUDAOptions instead of CUDATarget
        from triton.compiler.backends.cuda import CUDAOptions as CUDATarget  # type: ignore
    except ImportError:
        # As a last resort we fall back to None and rely on default compile target
        CUDATarget = None  # type: ignore

from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.contracts import CompileIn, CompileOut

from utils.logging_utils import get_logger

logger = get_logger("CompileAgent")


def _compile_kernel(payload: CompileIn) -> CompileOut:
    """Compile Triton kernel source to PTX and return path. Requires CUDA device."""
    import triton

    logger.info("Compiling kernel source (len=%d chars)", len(payload.kernel_src))

    # Save CompileIn payload to JSON
    try:
        input_dir = Path("tmp_compile_inputs")
        input_dir.mkdir(parents=True, exist_ok=True)
        ts_input_save = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Unique timestamp for this operation
        rand_suffix_input_save = random.randint(1000, 9999)
        input_filename = input_dir / f"compile_input_{ts_input_save}_{rand_suffix_input_save}.json"
        with open(input_filename, 'w') as f_in:
            f_in.write(payload.model_dump_json(indent=2))
        logger.info(f"Compile input (payload) saved to {input_filename.resolve()}")
    except Exception as e_save_in_payload:
        logger.error(f"Failed to save compile input payload to JSON: {e_save_in_payload}", exc_info=True)

    # Save the raw kernel_src to a .py file for direct inspection
    source_filename = None
    try:
        source_save_dir = Path("tmp_compile_kernel_sources")
        source_save_dir.mkdir(parents=True, exist_ok=True)
        # Use the same timestamp and random suffix as the payload save for easier correlation if needed
        # Or generate a new one if strict independence is preferred.
        # For simplicity, let's use a new one for this specific file log.
        ts_src_save = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rand_suffix_src_save = random.randint(1000, 9999)
        source_filename = source_save_dir / f"kernel_src_to_compile_{ts_src_save}_{rand_suffix_src_save}.py"
        with open(source_filename, 'w', encoding='utf-8') as f_src:
            f_src.write(payload.kernel_src)
        logger.info(f"Raw kernel_src (to be compiled) saved to {source_filename.resolve()}")
    except Exception as e_save_src:
        logger.error(f"Failed to save raw kernel_src to .py file: {e_save_src}", exc_info=True)
        source_filename = None

    # 1. Write the kernel source to a temporary python module
    tmp_py_file = None
    try:
        tmp_py_file = tempfile.NamedTemporaryFile(suffix="_kernel.py", delete=False, mode='w', encoding='utf-8')
        tmp_py_file.write(payload.kernel_src)
        tmp_py_file.flush()
        tmp_py_file.close()

        # 2. Load the module dynamically
        spec = importlib.util.spec_from_file_location("triton_user_kernel", tmp_py_file.name)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not get spec for module {tmp_py_file.name}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 3. Find the first Triton kernel (function with _is_triton_kernel attr or Autotuner)
        kernel_fn = None
        logger.debug("Scanning module for Triton kernels...")
        for v_name, v_obj in mod.__dict__.items(): # Iterate over name, object pairs
            logger.debug(f"  Found object: {v_name} = {type(v_obj)} (callable: {callable(v_obj)})")
            
            # Check for direct JITFunction
            if callable(v_obj) and isinstance(v_obj, JITFunction):
                kernel_fn = v_obj
                logger.info(f"Found @triton.jit kernel: {v_name} of type {type(v_obj)}")
                break
            # Check for Autotuner (which wraps JITFunction for @triton.autotune kernels)
            elif callable(v_obj) and Autotuner is not None and isinstance(v_obj, Autotuner) and hasattr(v_obj, 'fn'):
                logger.debug(f"  Object {v_name} has 'fn' attribute: {type(v_obj.fn)}")
                if isinstance(v_obj.fn, JITFunction):
                    kernel_fn = v_obj.fn  # Extract the underlying JITFunction from Autotuner
                    logger.info(f"Found @triton.autotune kernel: {v_name} of type {type(v_obj)}, extracted JITFunction")
                    break
            # Direct Autotuner instance (in some Triton versions it may not be callable or callable check can fail)
            elif Autotuner is not None and isinstance(v_obj, Autotuner) and hasattr(v_obj, 'fn'):
                logger.info(f"Found Autotuner instance for kernel: {v_name} – extracting wrapped JITFunction")
                kernel_fn = v_obj.fn
                break
            # Fallback: Some Triton versions may alias Autotuner under a different module path, causing
            # `isinstance(v_obj, Autotuner)` to fail even though the object behaves like one. If the object
            # is callable, exposes a `fn` attribute, and that attribute is a `JITFunction`, treat it as a
            # wrapped kernel and extract it.
            elif callable(v_obj) and hasattr(v_obj, 'fn'):
                possible_fn = getattr(v_obj, 'fn', None)
                # Accept if the wrapped object looks like a Triton JIT kernel (heuristic: has 'asm' or '_cache' or '_signature' attrs)
                if possible_fn is not None and (
                    isinstance(possible_fn, JITFunction)
                    or hasattr(possible_fn, "asm")
                    or hasattr(possible_fn, "compile")
                ):
                    kernel_fn = possible_fn  # Extract the underlying JITFunction (or equivalent)
                    logger.info(
                        f"Found Triton kernel via relaxed .fn detection: {v_name} of wrapper type {type(v_obj)}"
                    )
                    break
            # Check for other Triton-related objects
            elif hasattr(v_obj, '__name__') and 'triton' in str(type(v_obj)).lower():
                logger.debug(f"  Found Triton-related object: {v_name} = {type(v_obj)}")

        if kernel_fn is None:
            logger.error("No @triton.jit kernel found. Available objects in module:")
            for v_name, v_obj in mod.__dict__.items():
                logger.error(f"  {v_name}: {type(v_obj)}")
            raise ValueError("No @triton.jit kernel found in generated source")

        # ---------------- NEW: Prepare default meta-parameters ----------------
        import inspect
        meta_kwargs: dict[str, int] = {}
        sig = inspect.signature(kernel_fn.fn)  # type: ignore[attr-defined]
        for name, param in sig.parameters.items():
            # Skip the implicit *ptr and size arguments that will be provided at runtime.
            # We only need to provide compile-time constants (tl.constexpr) here.
            upper_name = name.upper()
            if "BLOCK" in upper_name:
                # Heuristic defaults: 128 for M/N tiles, 32 for K-tiles unless pattern suggests otherwise
                if "K" in upper_name:
                    meta_kwargs[name] = 32
                else:
                    meta_kwargs[name] = 128
            elif name in ("num_warps", "num_stages"):
                meta_kwargs[name] = 4 if name == "num_warps" else 2
        if meta_kwargs:
            logger.info("Supplying default meta-parameters for compilation: %s", meta_kwargs)
        # ---------------------------------------------------------------------

        # 4. Compile to PTX
        # Instantiate target descriptor when available (newer Triton). Otherwise rely on defaults.
        if CUDATarget is not None:
            # For Triton 2.3.1, determine the correct target tuple and use CUDAOptions
            import torch
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                target_tuple = ("cuda", capability[0] * 10 + capability[1])
                cuda_options = CUDATarget(num_warps=4, num_stages=3)  # type: ignore[call-arg]
                logger.info(f"Compiling for target: {target_tuple} with CUDAOptions")
            else:
                target_tuple = ("cuda", 80)  # Default to A100-like capability
                cuda_options = CUDATarget(num_warps=4, num_stages=3)  # type: ignore[call-arg]
                logger.info(f"CUDA not available, using default target: {target_tuple}")
            
            # Use ASTSource for compilation with meta-parameters
            try:
                from triton.compiler import ASTSource  # type: ignore
                import inspect

                sig_parts: list[str] = []
                param_names = list(inspect.signature(kernel_fn.fn).parameters.keys())  # type: ignore[attr-defined]

                for name in param_names:
                    if name.endswith("_ptr") or name.endswith("_PTR") or "ptr" in name.lower():
                        sig_parts.append("*fp32")  # assume 32-bit float pointer
                    else:
                        sig_parts.append("i32")  # default int32 for scalars/strides

                # Ensure order preserved
                signature_str = ",".join(sig_parts)

                # convert meta_kwargs to index→value dict based on param ordering
                const_map = {
                    (param_names.index(k),): v for k, v in meta_kwargs.items() if k in param_names
                }

                ast_src = ASTSource(kernel_fn, signature_str, const_map)
                ptx_kernel = triton.compile(ast_src, target=target_tuple, options=vars(cuda_options))
                ptx = ptx_kernel.asm["ptx"]  # type: ignore[index]
            except Exception as e_ast:
                logger.error("ASTSource compilation failed: %s", e_ast, exc_info=True)
                raise
        else:
            logger.info("CUDATarget class not found – compiling with default target settings")
            # For the default compilation path, use ASTSource without target
            try:
                from triton.compiler import ASTSource  # type: ignore
                import inspect
                
                sig_parts: list[str] = []
                param_names = list(inspect.signature(kernel_fn.fn).parameters.keys())  # type: ignore[attr-defined]
                
                for name in param_names:
                    if name.endswith("_ptr") or name.endswith("_PTR") or "ptr" in name.lower():
                        sig_parts.append("*fp32")  # assume 32-bit float pointer
                    else:
                        sig_parts.append("i32")  # default int32 for scalars/strides
                
                signature_str = ",".join(sig_parts)
                
                # convert meta_kwargs to index→value dict based on param ordering
                const_map = {
                    (param_names.index(k),): v for k, v in meta_kwargs.items() if k in param_names
                }
                
                ast_src = ASTSource(kernel_fn, signature_str, const_map)
                ptx_kernel = triton.compile(ast_src)
                ptx = ptx_kernel.asm["ptx"]  # type: ignore[index]
            except Exception as e_default:
                logger.error("Default ASTSource compilation failed: %s", e_default, exc_info=True)
                raise
        logger.debug("Triton compile finished, PTX length=%d", len(ptx))

        # 5. Persist PTX to file
        ptx_path_obj = Path(tempfile.gettempdir()) / f"{kernel_fn.__name__}.ptx"
        ptx_path_obj.write_text(ptx)
        logger.info("Kernel compiled successfully -> %s", ptx_path_obj)

        compile_out_obj = CompileOut(
            ok=True, 
            ptx_path=str(ptx_path_obj),
            source_file_path=str(source_filename) if source_filename else None
        )
        # Save CompileOut object to JSON (Success Case)
        try:
            output_dir = Path("tmp_compile_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            ts_out = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rand_suffix_out = random.randint(1000, 9999)
            output_filename = output_dir / f"compile_output_ok_{ts_out}_{rand_suffix_out}.json"
            with open(output_filename, 'w') as f_out:
                f_out.write(compile_out_obj.model_dump_json(indent=2))
            logger.info(f"Compile output (OK) saved to {output_filename.resolve()}")
        except Exception as e_save_out:
            logger.error(f"Failed to save compile output (OK) to JSON: {e_save_out}", exc_info=True)
        return compile_out_obj

    except Exception as e:
        log_message = str(e)
        logger.error("Compilation failed: %s", log_message)
        src_snippet = payload.kernel_src[:400] # Keep the short snippet for logging/quick view
        # Create CompileOut with both full source and the snippet
        compile_out_obj = CompileOut(
            ok=False, 
            log=log_message, 
            src_snippet_for_log=src_snippet, 
            full_kernel_src=payload.kernel_src,
            source_file_path=str(source_filename) if source_filename else None
        )
        # Save CompileOut object to JSON (Failure Case)
        try:
            output_dir = Path("tmp_compile_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            ts_out = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rand_suffix_out = random.randint(1000, 9999)
            output_filename = output_dir / f"compile_output_fail_{ts_out}_{rand_suffix_out}.json"
            with open(output_filename, 'w') as f_out:
                f_out.write(compile_out_obj.model_dump_json(indent=2))
            logger.info(f"Compile output (Fail) saved to {output_filename.resolve()}")
        except Exception as e_save_out:
            logger.error(f"Failed to save compile output (Fail) to JSON: {e_save_out}", exc_info=True)
        return compile_out_obj
    finally:
        if tmp_py_file and Path(tmp_py_file.name).exists():
            try:
                os.remove(tmp_py_file.name)
            except Exception as e_clean:
                logger.error(f"Failed to delete temporary compile file {tmp_py_file.name}: {e_clean}")

compile_tool = FunctionTool(_compile_kernel)


class CompileAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="compile",
            description="Compiles Triton kernel source into PTX binaries.",
            tools=[compile_tool]
        )

    async def compile(self, payload: CompileIn) -> CompileOut:
        return _compile_kernel(payload)
