"""Utility script to compile `kernel.py` into PTX.

This was originally written against Triton ≥3 which exposes
 `triton.compiler.frontend.convert_python_to_ttir` and
 `triton.compiler.codegen.compile_ttir_to_ptx` helpers.

Those symbols are not available in the Triton 2.x series.  To keep the
project working after we pinned the dependency to Triton 2, we fall back
to the (stable) public `triton.compile` API when the newer helpers are
missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import triton


KERNEL_SRC_PATH = Path("tmp_compile_kernel_sources/kernel_src_to_compile_20250603_094604_000958_5688.py")

# ---------------------------------------------------------------------------
# First, try to use the newer helper APIs (Triton 3).  If they are not
# present, we gracefully degrade to a Triton 2 compatible path.
# ---------------------------------------------------------------------------

try:
    # Triton ≥3 style import
    from triton.compiler import frontend, codegen  # type: ignore

    src_text = KERNEL_SRC_PATH.read_text()
    ttir = frontend.convert_python_to_ttir(
        src_text, defines={"BLOCK_M": 128, "BLOCK_N": 128}
    )
    ptx = codegen.compile_ttir_to_ptx(ttir, arch="sm_90")

except (ModuleNotFoundError, ImportError):
    # ---------------------------------------------------------------------
    # Triton 2 fallback: dynamically import the kernel file, locate the
    # first @triton.jit function, and compile it via `triton.compile`.
    # ---------------------------------------------------------------------
    from tempfile import TemporaryDirectory

    # Dynamically load the user's kernel module --------------------------------
    spec = importlib.util.spec_from_file_location("user_kernel", str(KERNEL_SRC_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {KERNEL_SRC_PATH} as a Python module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    # Locate the first Triton JIT function -------------------------------------
    try:
        from triton.runtime.jit import JITFunction  # type: ignore
    except ImportError:
        from triton import JITFunction  # type: ignore

    kernel_fn = None
    for obj in module.__dict__.values():
        if isinstance(obj, JITFunction):
            kernel_fn = obj
            break

    if kernel_fn is None:
        raise RuntimeError("No @triton.jit kernel found in 'kernel.py'.")

    # Compile to PTX -----------------------------------------------------------
    # In Triton 2 the compile() helper does not accept `constants` nor
    # arbitrary meta-parameter kwargs.  We therefore progressively degrade the
    # call signature until we find one that works.

    compile_constants = {"BLOCK_M": 128, "BLOCK_N": 128}

    # Build a list of attempts with progressively simpler argument sets.
    compile_attempts: list[dict] = []

    # Triton >=3 signature
    compile_attempts.append(dict(constants=compile_constants, target="sm_90"))

    # Triton 2.1-2.2 accepted meta-params as plain kwargs
    compile_attempts.append({**compile_constants, "target": "sm_90"})

    # For Triton 2.3 passing a string target triggers an assertion. If we have
    # access to the `GPUTarget` helper we build one. Otherwise drop the target.
    try:
        from triton.backends.compiler import GPUTarget  # type: ignore

        compile_attempts.append({"target": GPUTarget("cuda", 90, 32)})
    except Exception:
        # Fallbacks without explicit target
        pass

    # Triton 2 expects an ASTSource object; we can build one and compile it.
    try:
        from triton.compiler import ASTSource  # type: ignore

        # signature must include *all* parameters, compile-time constants
        # are provided via an index->value dict.
        sig_parts = [
            "*fp32", "*fp32", "*fp32",            # 0-2 pointers
            *["i32"] * 12                           # 3-14 ints (M,N,K and strides)
        ]
        # total so far 15 elements (0-14). Add five unused placeholder types
        # for the constexpr parameters so that positions align.
        sig_parts.extend(["i32"] * 5)  # BLOCK_M,N,K,num_stages, num_warps
        sig = ",".join(sig_parts)

        # Map index → value for compile-time constants
        const_map = {
            15: 128,   # BLOCK_M
            16: 128,   # BLOCK_N (will be overridden maybe) but same
            17: 32,    # BLOCK_K
            18: 3,     # num_stages
            19: 8,     # num_warps
        }
        ast_src = ASTSource(kernel_fn, sig, const_map)
        compile_attempts.append({"src_override": ast_src})
    except Exception:
        pass

    # Finally, attempt with no extra kwargs
    compile_attempts.append({})

    last_err: Exception | None = None
    for kwargs in compile_attempts:
        try:
            if "src_override" in kwargs:
                src = kwargs.pop("src_override")  # type: ignore[var-annotated]
                ptx_kernel = triton.compile(src, **kwargs)  # returns CompiledKernel
                ptx = ptx_kernel.asm["ptx"]  # type: ignore[index]
            else:
                ptx = triton.compile(kernel_fn, **kwargs)  # type: ignore[arg-type]
            break  # success
        except (TypeError, AssertionError, RuntimeError) as exc:
            last_err = exc
    else:
        # If none of the attempts succeeded, raise the last error.
        raise last_err  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Persist the resulting PTX to disk.
# ---------------------------------------------------------------------------

Path("kernel.ptx").write_text(ptx)
print("PTX written to kernel.ptx (length =", len(ptx), ")")