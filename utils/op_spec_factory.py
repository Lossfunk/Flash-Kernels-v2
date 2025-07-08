from __future__ import annotations

"""Utility helpers to build OpSpec objects from arbitrary user-supplied code.

These functions allow the MCP layer to turn either a raw PyTorch function body
or a full ``nn.Module`` source file into the :class:`agents.contracts.OpSpec`
structure that ``KernelPipelineAgent`` expects.

Why separate from ``utils.kernelbench``?
---------------------------------------
The original ``utils.kernelbench`` helpers assume KernelBench problem layout
(levels, ids, etc.).  In a real chat setting the user will paste arbitrary code
without those conventions.  These helpers keep the core spec-building logic
re-usable without KernelBench baggage.
"""

import inspect
import textwrap
import types
import uuid
from typing import Any, Dict, List

import torch

from agents.contracts import OpSpec, TensorSpec

__all__ = [
    "from_src",
    "from_module",
]


def _tensor_to_spec(t: torch.Tensor | float | int) -> TensorSpec:
    """Convert a tensor or scalar into a TensorSpec."""
    if isinstance(t, torch.Tensor):
        return TensorSpec(shape=list(t.shape), dtype=str(t.dtype).replace("torch.", ""))
    if isinstance(t, float):
        return TensorSpec(shape=[], dtype="float32")
    if isinstance(t, int):
        return TensorSpec(shape=[], dtype="int64")
    raise TypeError(f"Unsupported input type {type(t)}. Expected tensor, int or float.")


def _extract_forward_from_src(module_src: str) -> str:
    """Extract the forward method source from a module source string."""
    import ast
    
    try:
        # Parse the module source into an AST
        tree = ast.parse(textwrap.dedent(module_src))
        
        # Find the Model class
        model_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                model_class = node
                break
        
        if model_class is None:
            raise ValueError("No 'Model' class found in module source")
        
        # Find the forward method
        forward_method = None
        for node in model_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                forward_method = node
                break
        
        if forward_method is None:
            raise ValueError("No 'forward' method found in Model class")
        
        # Convert the AST node back to source code
        import ast
        return ast.unparse(forward_method)
        
    except Exception as e:
        # Fallback: try to extract forward method using simple string parsing
        lines = module_src.split('\n')
        forward_lines = []
        in_forward = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def forward('):
                in_forward = True
                indent_level = len(line) - len(line.lstrip())
                forward_lines.append(line)
            elif in_forward:
                if line.strip() == '':
                    forward_lines.append(line)
                elif len(line) - len(line.lstrip()) > indent_level:
                    forward_lines.append(line)
                else:
                    break
        
        if not forward_lines:
            raise ValueError(f"Could not extract forward method from source: {e}")
        
        return '\n'.join(forward_lines)


# -----------------------------------------------------------------------------
#  Public factory helpers
# -----------------------------------------------------------------------------

def from_src(
    pytorch_src: str,
    inputs: List[torch.Tensor | int | float],
    *,
    op_params: Dict[str, Any] | None = None,
    problem_id: int | None = None,
    level: int | None = None,
) -> OpSpec:
    """Build an OpSpec from a *function body* or full function source.

    Parameters
    ----------
    pytorch_src:
        The source code implementing the computation in plain PyTorch.  This can
        be a full ``def forward(...):`` function or just the body.  It is
        preserved verbatim inside the OpSpec so downstream agents can feed it
        to the LLM.
    inputs:
        Concrete sample inputs used only to capture shapes/dtypes; the tensors
        need not hold meaningful values.
    op_params, problem_id, level:
        Optional metadata carried through for compatibility with existing
        KernelPipelineAgent logging.
    """
    input_specs = [_tensor_to_spec(t) for t in inputs]

    return OpSpec(
        problem_id=problem_id or -1,
        level=level or 0,
        pytorch_src=textwrap.dedent(pytorch_src),
        input_specs=input_specs,
        op_params=op_params,
    )


def from_module(
    module_src: str,
    init_inputs: List[Any] | None,
    sample_inputs: List[torch.Tensor | int | float],
    *,
    op_params: Dict[str, Any] | None = None,
    problem_id: int | None = None,
    level: int | None = None,
) -> OpSpec:
    """Build an OpSpec representing *entire* ``nn.Module.forward``.

    This compiles the provided ``module_src`` string, instantiates the ``Model``
    class it must contain, extracts its ``forward`` source code, and finally
    delegates to :func:`from_src`.

    Notes
    -----
    * ``exec`` is inherently unsafe with untrusted code.  In production, run
      inside a container with restricted privileges.
    * We assume the top-level symbol is named ``Model`` following the
      KernelBench convention.  Adjust if you want to support arbitrary class
      names.
    """
    # 1. Materialise the module dynamically so inspect.getsource works.
    module_name = f"user_mod_{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(module_name)
    exec(textwrap.dedent(module_src), mod.__dict__)

    if "Model" not in mod.__dict__:
        raise AttributeError("Provided module_src must define a class named 'Model'.")

    ModelCls = mod.__dict__["Model"]

    # 2. Instantiate so forward is bound correctly (some models build layers in __init__).
    model_instance = ModelCls(*((init_inputs or [])))  # noqa: B010
    # Keep model on CPU for spec building; it will be moved to CUDA downstream.

    # 3. Extract the forward source from the original module_src string.
    # Since the model was created dynamically, inspect.getsource() won't work.
    # We need to parse the forward method from the original source.
    forward_src = _extract_forward_from_src(module_src)

    input_specs = [_tensor_to_spec(t) for t in sample_inputs]

    return OpSpec(
        problem_id=problem_id or -1,
        level=level or 0,
        pytorch_src=textwrap.dedent(forward_src),
        input_specs=input_specs,
        op_params=op_params,
        module_src=module_src,  # Include full module source for profiling
        init_inputs=init_inputs  # Include init inputs for model construction
    ) 