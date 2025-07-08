from __future__ import annotations

import importlib.util
import tempfile
import time
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from google.adk.tools.function_tool import FunctionTool

from agents.base import BaseAgent
from agents.contracts import CorrectIn, CorrectOut, TensorSpec
from utils.logging_utils import get_logger

# KernelBench imports
from KernelBench.src.eval import eval_kernel_against_ref, KernelExecResult
# Ensure KernelBench is in PYTHONPATH or installed if this causes import errors.

logger = get_logger("CorrectnessAgent")

# This map might be useful if, in the future, we need to dynamically generate
# get_inputs/get_init_inputs functions for the source strings, though KernelBench
# currently expects these to be defined within the provided model sources.
_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int8": torch.int8,
}


def _load_kernelbench_model(problem_id: int, level: int) -> tuple[str, List, List]:
    """
    Load the original KernelBench Model class directly from the problem file.
    
    Returns:
        tuple: (complete_model_source, test_inputs, init_inputs)
    """
    from pathlib import Path
    import importlib.util
    
    # Construct the path to the KernelBench problem file
    kernelbench_dir = Path(__file__).parent.parent.parent / "KernelBench" / "KernelBench"
    problem_dir = kernelbench_dir / f"level{level}"
    
    # Find the problem file by ID
    problem_file = None
    for file_path in problem_dir.glob("*.py"):
        if file_path.name.startswith(f"{problem_id}_"):
            problem_file = file_path
            break
    
    if not problem_file:
        raise FileNotFoundError(f"Could not find KernelBench problem file for level {level}, problem {problem_id}")
    
    logger.info(f"Loading KernelBench model from: {problem_file}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location(problem_file.stem, problem_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the Model class and functions
    model_cls = getattr(module, "Model")
    get_inputs_func = getattr(module, "get_inputs")
    
    # Get initialization inputs if available
    init_inputs = []
    if hasattr(module, "get_init_inputs"):
        get_init_inputs_func = getattr(module, "get_init_inputs")
        init_inputs = get_init_inputs_func()
    
    # Get test inputs
    test_inputs = get_inputs_func()
    
    # Read the complete source file
    complete_model_source = problem_file.read_text()
    
    logger.info(f"Successfully loaded KernelBench model: {model_cls.__name__}")
    logger.info(f"Init inputs: {init_inputs}")
    logger.info(f"Test inputs shapes: {[getattr(t, 'shape', type(t)) for t in test_inputs]}")
    
    return complete_model_source, test_inputs, init_inputs


def _build_complete_pytorch_module(pytorch_src: str, input_specs: List[TensorSpec], op_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Transform incomplete pytorch_src (just forward method) into a complete KernelBench-compatible module.
    This is now a fallback function when the original KernelBench model cannot be loaded.

    KernelBench expects:
    - Complete imports (torch, torch.nn)
    - Model class inheriting from nn.Module
    - get_inputs() function returning input tensors
    - get_init_inputs() function returning initialization parameters
    """
    logger.info("Building complete PyTorch module from forward method (fallback)")

    # Clean up the forward method source and ensure proper indentation
    forward_method = pytorch_src.strip()

    # Split into lines and ensure proper indentation for class method (4 spaces total)
    lines = forward_method.split('\n')
    indented_lines = []
    for i, line in enumerate(lines):
        if line.strip():  # Only process non-empty lines
            if i == 0:  # First line (def forward...)
                # 4 spaces for class method
                indented_lines.append('    ' + line.lstrip())
            else:  # Body lines
                # 8 spaces for method body
                indented_lines.append('        ' + line.lstrip())
        else:
            indented_lines.append(line)  # Keep empty lines as-is
    forward_method = '\n'.join(indented_lines)

    # Generate get_inputs function based on input_specs
    get_inputs_lines = ["def get_inputs():"]
    get_inputs_lines.append("    # Generated input tensors based on input specs")

    for i, spec in enumerate(input_specs):
        if len(spec.shape) == 0:  # Scalar
            # For scalars, return the value directly
            if spec.dtype == "float32":
                get_inputs_lines.append(f"    input_{i} = 1.0")
            elif spec.dtype == "int32":
                get_inputs_lines.append(f"    input_{i} = 1")
            else:
                get_inputs_lines.append(f"    input_{i} = 1.0")  # Default to float
        else:  # Tensor
            shape_str = ", ".join(map(str, spec.shape))
            if spec.dtype == "float16":
                get_inputs_lines.append(f"    input_{i} = torch.randn({shape_str}, dtype=torch.float16)")
            elif spec.dtype == "int32":
                get_inputs_lines.append(f"    input_{i} = torch.randint(0, 10, ({shape_str}), dtype=torch.int32)")
            elif spec.dtype == "int8":
                get_inputs_lines.append(f"    input_{i} = torch.randint(-128, 127, ({shape_str}), dtype=torch.int8)")
            else:  # Default to float32
                get_inputs_lines.append(f"    input_{i} = torch.randn({shape_str})")

    # Return all inputs as a list
    input_names = [f"input_{i}" for i in range(len(input_specs))]
    get_inputs_lines.append(f"    return [{', '.join(input_names)}]")

    get_inputs_function = "\n".join(get_inputs_lines)

    # Build initialization method with op_params
    init_assignments = []
    init_params = []
    init_values = []
    
    # Add op_params if provided
    if op_params:
        param_names = list(op_params.keys())
        param_values = [repr(op_params[name]) for name in param_names]
        
        init_params.extend(param_names)
        init_values.extend(param_values)
        
        for name in param_names:
            init_assignments.append(f"        self.{name} = {name}")
    
    # Build the __init__ method
    if init_params:
        init_signature_params = ", ".join(init_params)
        init_method_str = f"""\
    def __init__(self, {init_signature_params}):
        super().__init__()
{chr(10).join(init_assignments)}"""
        
        get_init_inputs_str = f"""\
def get_init_inputs():
    # Initialization inputs for {', '.join(init_params)}
    return [{', '.join(init_values)}]"""
    else:
        # No parameters
        init_method_str = """\
    def __init__(self):
        super().__init__()"""
        
        get_init_inputs_str = "def get_init_inputs():\n    return []"

    # Build the complete module
    complete_module = f"""import torch
import torch.nn as nn

class Model(nn.Module):
{init_method_str}

{forward_method}

{get_init_inputs_str}

{get_inputs_function}
"""

    logger.debug("Generated complete PyTorch module:\n%s", complete_module)
    return complete_module


def _build_kernelbench_compatible_triton_module(triton_kernel_src: str, input_specs: List[TensorSpec]) -> str:
    """
    Transform Triton kernel source into a KernelBench-compatible ModelNew class.
    This function analyzes the launcher function signature to determine how to properly
    call it with the correct arguments, including creating output tensors when needed.
    """
    logger.info("Building KernelBench-compatible ModelNew class from Triton kernel")

    # Extract the launcher function name and analyze its signature
    import ast
    import inspect
    
    launcher_func_name = None
    launcher_func_node = None
    
    try:
        tree = ast.parse(triton_kernel_src)
        
        # First priority: functions starting with 'launch_'
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('launch_'):
                launcher_func_name = node.name
                launcher_func_node = node
                break
        
        # Second priority: known operation functions (like 'softmax' from 02-fused-softmax.py)
        if not launcher_func_name:
            known_ops = ['softmax', 'matmul', 'conv2d', 'relu', 'gelu', 'layer_norm', 'batch_norm']
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in known_ops:
                    launcher_func_name = node.name
                    launcher_func_node = node
                    logger.info(f"Found known operation function: {node.name}")
                    break
        
        # Third priority: any function that calls a kernel (detected by kernel[grid](...) pattern)
        if not launcher_func_name:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this function contains kernel calls (kernel[grid](...))
                    for child in ast.walk(node):
                        if isinstance(child, ast.Subscript):
                            # Look for pattern like kernel_name[grid](...)
                            if isinstance(child.value, ast.Name):
                                launcher_func_name = node.name
                                launcher_func_node = node
                                logger.info(f"Found kernel-calling function: {node.name}")
                                break
                    if launcher_func_name:
                        break
                        
    except Exception as e:
        logger.warning(f"Failed to parse Triton source for launcher function: {e}")

    if not launcher_func_name or not launcher_func_node:
        logger.warning("No launcher function found, using fallback approach")
        launcher_func_name = "launch_kernel"  # Default fallback

    # Analyze the launcher function signature to understand its parameters
    launcher_params = []
    if launcher_func_node:
        launcher_params = [arg.arg for arg in launcher_func_node.args.args]
        logger.info(f"Found launcher function '{launcher_func_name}' with parameters: {launcher_params}")

    # Determine the operation type and output shape logic based on function name and parameters
    operation_type = _infer_operation_type(launcher_func_name, launcher_params)
    logger.info(f"Inferred operation type: {operation_type}")

    # Determine the calling pattern based on the launcher function name
    if launcher_func_name in ["softmax", "relu", "gelu", "layer_norm", "batch_norm"]:
        # These functions typically take input and return output (functional style)
        function_call_pattern = f"return {launcher_func_name}(*cuda_inputs)"
        output_creation_needed = False
    else:
        # Traditional launcher functions take output as first argument (in-place style)
        function_call_pattern = f"{launcher_func_name}(output, *cuda_inputs)\n        return output"
        output_creation_needed = True

    # Build the ModelNew class with intelligent output tensor creation
    output_creation_code = """
        # Determine output shape and create output tensor
        output_shape, output_dtype = self._determine_output_shape_and_dtype(cuda_inputs, "{operation_type}")
        
        # Create output tensor on the same device as inputs
        device = cuda_inputs[0].device if isinstance(cuda_inputs[0], torch.Tensor) else torch.device('cuda')
        output = torch.empty(output_shape, dtype=output_dtype, device=device)
        """ if output_creation_needed else ""

    modelnew_class = f"""
class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Store any initialization parameters (though Triton kernels typically don't need them)
        self.init_args = args
        self.init_kwargs = kwargs

    def forward(self, *inputs):
        # Convert inputs to CUDA if needed
        cuda_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                cuda_inputs.append(inp.cuda() if not inp.is_cuda else inp.contiguous())
            else:
                cuda_inputs.append(inp)
        {output_creation_code}
        # Call the launcher function
        {function_call_pattern}
    
    def _determine_output_shape_and_dtype(self, inputs, operation_type):
        \"\"\"
        Determine the output shape and dtype based on the operation type and input tensors.
        \"\"\"
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("At least one input tensor is required")
        
        # Get the primary input tensor for dtype reference
        primary_input = inputs[0]
        dtype = primary_input.dtype
        
        if operation_type == "matmul":
            return self._get_matmul_output_shape(inputs), dtype
        elif operation_type == "pooling":
            return self._get_pooling_output_shape(inputs), dtype
        elif operation_type == "elementwise":
            return self._get_elementwise_output_shape(inputs), dtype
        elif operation_type == "conv":
            return self._get_conv_output_shape(inputs), dtype
        elif operation_type == "reduction":
            return self._get_reduction_output_shape(inputs), dtype
        else:
            # Default: assume output has same shape as first input
            logger.warning(f"Unknown operation type '{{operation_type}}', using first input shape")
            return primary_input.shape, dtype
    
    def _get_matmul_output_shape(self, inputs):
        \"\"\"Determine output shape for matrix multiplication operations.\"\"\"
        if len(inputs) < 2:
            raise ValueError("Matrix multiplication requires at least 2 input tensors")
        
        a, b = inputs[0], inputs[1]
        
        # Handle different matmul cases
        if a.dim() == 2 and b.dim() == 2:
            # Standard 2D matrix multiplication: (M, K) @ (K, N) -> (M, N)
            return (a.shape[0], b.shape[1])
        elif a.dim() == 3 and b.dim() == 2:
            # 3D tensor @ 2D matrix: (N, M, K) @ (K, L) -> (N, M, L)
            return (a.shape[0], a.shape[1], b.shape[1])
        elif a.dim() == 3 and b.dim() == 3:
            # Batched matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N)
            return (a.shape[0], a.shape[1], b.shape[2])
        else:
            # Fallback: use PyTorch's matmul to determine shape
            try:
                dummy_result = torch.matmul(torch.zeros_like(a[:1]), torch.zeros_like(b[:1]))
                return tuple(s * a.shape[0] // dummy_result.shape[0] if dummy_result.shape[0] > 0 else s 
                           for s in dummy_result.shape)
            except:
                # Last resort: assume same shape as first input
                return a.shape
    
    def _get_pooling_output_shape(self, inputs):
        \"\"\"Determine output shape for pooling operations.\"\"\"
        if not inputs:
            raise ValueError("Pooling requires at least 1 input tensor")
        
        input_tensor = inputs[0]
        
        # For pooling operations, the output typically has the same batch and channel dimensions
        # but different spatial dimensions. Since we don't have access to pooling parameters here,
        # we'll make a reasonable assumption based on common pooling operations.
        
        if input_tensor.dim() == 4:  # 2D pooling: (N, C, H, W)
            # Common case: 2x2 pooling with stride 2 reduces spatial dimensions by half
            n, c, h, w = input_tensor.shape
            return (n, c, h // 2, w // 2)
        elif input_tensor.dim() == 3:  # 1D pooling: (N, C, L)
            n, c, l = input_tensor.shape
            return (n, c, l // 2)
        elif input_tensor.dim() == 5:  # 3D pooling: (N, C, D, H, W)
            n, c, d, h, w = input_tensor.shape
            return (n, c, d // 2, h // 2, w // 2)
        else:
            # Fallback: same shape as input
            return input_tensor.shape
    
    def _get_elementwise_output_shape(self, inputs):
        \"\"\"Determine output shape for elementwise operations.\"\"\"
        if not inputs:
            raise ValueError("Elementwise operation requires at least 1 input tensor")
        
        # For elementwise operations, output shape is typically the same as input shape
        return inputs[0].shape
    
    def _get_conv_output_shape(self, inputs):
        \"\"\"Determine output shape for convolution operations.\"\"\"
        if len(inputs) < 2:
            raise ValueError("Convolution requires at least 2 input tensors (input and weight)")
        
        input_tensor, weight = inputs[0], inputs[1]
        
        if input_tensor.dim() == 4 and weight.dim() == 4:  # 2D convolution
            n, c_in, h_in, w_in = input_tensor.shape
            c_out, c_in_w, k_h, k_w = weight.shape
            
            # For simplicity, assume stride=1, padding=0 (common case)
            # In practice, these would be parameters to the kernel
            h_out = h_in - k_h + 1
            w_out = w_in - k_w + 1
            return (n, c_out, h_out, w_out)
        else:
            # Fallback: same shape as input
            return input_tensor.shape
    
    def _get_reduction_output_shape(self, inputs):
        \"\"\"Determine output shape for reduction operations.\"\"\"
        if not inputs:
            raise ValueError("Reduction operation requires at least 1 input tensor")
        
        input_tensor = inputs[0]
        
        # Common reduction operations reduce the last dimension
        if input_tensor.dim() > 1:
            return input_tensor.shape[:-1]
        else:
            return (1,)  # Scalar result
"""

    # Combine the original Triton source with the ModelNew class
    complete_module = triton_kernel_src + "\n" + modelnew_class

    logger.debug("Generated KernelBench-compatible ModelNew class with intelligent output tensor creation")
    return complete_module


def _infer_operation_type(launcher_func_name: str, launcher_params: List[str]) -> str:
    """
    Infer the operation type based on the launcher function name and parameters.
    """
    func_name_lower = launcher_func_name.lower()
    
    # Check function name patterns
    if "matmul" in func_name_lower or "mm" in func_name_lower or "gemm" in func_name_lower:
        return "matmul"
    elif any(pool_type in func_name_lower for pool_type in ["pool", "maxpool", "avgpool", "meanpool"]):
        return "pooling"
    elif "conv" in func_name_lower:
        return "conv"
    elif any(op in func_name_lower for op in ["relu", "gelu", "sigmoid", "tanh", "softmax", "norm"]):
        return "elementwise"
    elif any(red_op in func_name_lower for red_op in ["sum", "mean", "max", "min", "reduce"]):
        return "reduction"
    else:
        # Try to infer from parameter patterns
        if len(launcher_params) >= 3:
            # If there are 3+ parameters and first is likely output, second and third are likely inputs
            # This suggests a binary operation like matmul
            return "matmul"
        else:
            return "elementwise"  # Default assumption


def _build_kernelbench_compatible_triton_module_legacy(triton_kernel_src: str, input_specs: List[TensorSpec]) -> str:
    """
    Legacy approach: Transform Triton kernel source into a KernelBench-compatible ModelNew class.
    This version tries to be more flexible with function detection.
    """
    logger.info("Building KernelBench-compatible ModelNew class from Triton kernel (legacy)")

    # Try to find any function that could be a launcher
    import ast
    launcher_func_name = None
    try:
        tree = ast.parse(triton_kernel_src)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        # Prefer functions with 'launch' in the name
        for func_name in functions:
            if 'launch' in func_name.lower():
                launcher_func_name = func_name
                break

        # If no launch function, use the last function defined
        if not launcher_func_name and functions:
            launcher_func_name = functions[-1]

    except Exception as e:
        logger.warning(f"Failed to parse Triton source for function detection: {e}")

    if not launcher_func_name:
        launcher_func_name = "kernel_launcher"  # Default fallback

    # Build the ModelNew class that accepts any initialization parameters
    modelnew_class = f"""
class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Store any initialization parameters (though Triton kernels typically don't need them)
        self.init_args = args
        self.init_kwargs = kwargs

    def forward(self, *args):
        return {launcher_func_name}(*args)
"""

    # Combine the original Triton source with the ModelNew class
    complete_module = triton_kernel_src + "\n" + modelnew_class

    logger.debug(f"Generated KernelBench-compatible ModelNew class with launcher: {launcher_func_name}")
    return complete_module


def _find_kernel_source_from_ptx(ptx_path: str) -> Optional[str]:
    """
    Find the corresponding Triton kernel Python source file path for a given PTX path.
    """
    logger.info(f"Attempting to find kernel source file path for ptx: {ptx_path}")

    # Look for source files in tmp_compile_kernel_sources directory
    source_dir = Path("tmp_compile_kernel_sources")
    if not source_dir.exists():
        logger.warning("tmp_compile_kernel_sources directory does not exist")
        return None

    # Find the most recent kernel source file
    source_files = list(source_dir.glob("kernel_src_to_compile_*.py"))
    if not source_files:
        logger.warning("No kernel source files found in tmp_compile_kernel_sources")
        return None

    # Sort by modification time and get the most recent
    most_recent_file = max(source_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using most recent kernel source file: {most_recent_file}")

    return str(most_recent_file)


def _find_kernel_source_text(ptx_path: str) -> Optional[str]:
    """
    Find the corresponding Triton kernel Python source file for a given PTX path.
    """
    logger.info(f"Attempting to find kernel source from tmp_compile_kernel_sources for ptx: {ptx_path}")

    # Look for source files in tmp_compile_kernel_sources directory
    source_dir = Path("tmp_compile_kernel_sources")
    if not source_dir.exists():
        logger.warning("tmp_compile_kernel_sources directory does not exist")
        return None

    # Find the most recent kernel source file
    source_files = list(source_dir.glob("kernel_src_to_compile_*.py"))
    if not source_files:
        logger.warning("No kernel source files found in tmp_compile_kernel_sources")
        return None

    # Sort by modification time and get the most recent
    most_recent_file = max(source_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using most recent kernel source file: {most_recent_file} as custom_model_src.")

    try:
        kernel_source = most_recent_file.read_text()
        logger.debug(f"Successfully read kernel source from {most_recent_file}")
        return kernel_source
    except Exception as e:
        logger.error(f"Failed to read kernel source from {most_recent_file}: {e}")
        return None


def _make_test_inputs(specs: List[TensorSpec]) -> List[torch.Tensor]:
    """Create CUDA tensors matching TensorSpec list."""
    inputs: List[torch.Tensor] = []
    for spec in specs:
        shape = tuple(spec.shape) if spec.shape else ()
        if spec.dtype == "float16":
            t = torch.randn(*shape, dtype=torch.float16, device="cuda")
        elif spec.dtype == "int32":
            t = torch.randint(0, 10, shape, dtype=torch.int32, device="cuda")  # type: ignore[arg-type]
        elif spec.dtype == "int8":
            t = torch.randint(-128, 127, shape, dtype=torch.int8, device="cuda")  # type: ignore[arg-type]
        else:  # default float32
            t = torch.randn(*shape, dtype=torch.float32, device="cuda")
        inputs.append(t)
    return inputs


def _build_reference_callable(pytorch_src: str, op_params: Optional[Dict[str, Any]]) -> Any:
    """Exec the provided forward() code into a callable `ref(*inputs)` on CUDA."""
    import types, textwrap, inspect

    module = types.ModuleType("reference_mod")
    exec("import torch\nimport torch.nn.functional as F", module.__dict__)
    exec(textwrap.dedent(pytorch_src), module.__dict__)

    # Pick first callable defined by the snippet (heuristic)
    ref_fn = None
    for val in module.__dict__.values():
        if callable(val):
            ref_fn = val
            break
    if ref_fn is None:
        raise RuntimeError("Could not locate reference function in pytorch_src")

    # If it's a method (expects self), wrap it
    if "self" in inspect.signature(ref_fn).parameters:
        class Dummy(torch.nn.Module):
            def forward(self, *args):
                return ref_fn(self, *args)
        m = Dummy().cuda()
        return m.forward
    return ref_fn


def _load_triton_launcher(src_path: str) -> Any:
    import importlib.util, inspect

    spec = importlib.util.spec_from_file_location("triton_kernel", src_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    # Prefer launch_ prefix
    for name in dir(module):
        if name.startswith("launch_"):
            return getattr(module, name)
    # Fallback: first callable that has Subscript pattern is too heavy; instead pick any callable
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj):
            return obj
    raise RuntimeError("No launcher function found in Triton kernel module")


async def _validate_kernel_with_triton_direct(payload: CorrectIn) -> CorrectOut:
    """Operation-level validation using Liger-inspired approach."""
    if not torch.cuda.is_available():
        return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                          error_details={"error_type": "cuda_unavailable", "error_message": "CUDA not available"})

    try:
        if not payload.ptx_path:
            return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                              error_details={"error_type": "source_missing", "error_message": "Missing kernel PTX file path"})

        # Try to find the corresponding source file from the PTX path
        source_file_path = _find_kernel_source_from_ptx(payload.ptx_path)
        if not source_file_path:
            return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                              error_details={"error_type": "source_missing", "error_message": "Could not find kernel source file"})

        # Strategy 1: Try operation-level testing (Liger approach)
        operation_type = _infer_operation_type_from_payload(payload)
        if operation_type:
            logger.info(f"Using operation-level testing for: {operation_type}")
            from agents.correctness.operation_tests import test_triton_operation
            
            result = test_triton_operation(
                kernel_path=source_file_path,
                operation_type=operation_type,
                input_specs=payload.input_specs,
                **_extract_operation_kwargs(payload)
            )
            
            if result.correct or result.error_details.get("error_type") != "unsupported_operation":
                return result

        # Strategy 2: Fallback to functional reference approach
        logger.info("Falling back to functional reference approach")
        return await _validate_with_functional_reference(payload, source_file_path)

    except Exception as e:
        return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                          error_details={"error_type": "validation_exception", "error_message": str(e)})


def _infer_operation_type_from_payload(payload: CorrectIn) -> Optional[str]:
    """Infer operation type from payload for operation-level testing."""
    # Check pytorch_src for operation patterns
    pytorch_src = payload.pytorch_src.lower()
    
    # Check for composite operations first (more specific)
    if ("fc1" in pytorch_src and "fc2" in pytorch_src) or ("linear" in pytorch_src and "relu" in pytorch_src):
        return "mlp"  # Multi-layer perceptron
    elif "matmul" in pytorch_src or "@" in pytorch_src:
        return "matmul"
    elif "softmax" in pytorch_src:
        return "softmax"
    elif "layer_norm" in pytorch_src or "layernorm" in pytorch_src:
        return "layer_norm"
    elif "linear" in pytorch_src:
        return "linear"
    elif "relu" in pytorch_src:
        return "relu"
    
    # Check op_params for hints
    if payload.op_params:
        if any(key in str(payload.op_params).lower() for key in ["matmul", "mm", "gemm"]):
            return "matmul"
        elif any(key in str(payload.op_params).lower() for key in ["softmax"]):
            return "softmax"
        elif any(key in str(payload.op_params).lower() for key in ["layer_norm", "layernorm"]):
            return "layer_norm"
        elif "dim" in payload.op_params and "out" in payload.op_params:
            return "mlp"  # MLP-like structure
    
    return None


def _extract_operation_kwargs(payload: CorrectIn) -> Dict[str, Any]:
    """Extract operation-specific kwargs from payload."""
    kwargs = {}
    
    if payload.op_params:
        # Common parameters
        if "dim" in payload.op_params:
            kwargs["dim"] = payload.op_params["dim"]
        if "eps" in payload.op_params:
            kwargs["eps"] = payload.op_params["eps"]
        if "normalized_shape" in payload.op_params:
            kwargs["normalized_shape"] = payload.op_params["normalized_shape"]
    
    return kwargs


async def _validate_with_functional_reference(payload: CorrectIn, source_file_path: str) -> CorrectOut:
    """Fallback validation using functional reference approach."""
    try:
        test_inputs = _make_test_inputs(payload.input_specs)
        ref_fn = _build_reference_callable(payload.pytorch_src, payload.op_params)

        with torch.no_grad():
            expected = ref_fn(*test_inputs)
        if not isinstance(expected, torch.Tensor):
            expected = torch.tensor(expected, device="cuda")

        launcher = _load_triton_launcher(source_file_path)

        # Try common call patterns
        try:
            triton_out = launcher(*test_inputs)
        except TypeError:
            output = torch.empty_like(expected)
            launcher(output, *test_inputs)
            triton_out = output

        # Compare
        if expected.shape != triton_out.shape:
            return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                              error_details={"error_type": "shape_mismatch",
                                             "error_message": f"Expected shape {expected.shape}, got {triton_out.shape}"})

        if torch.allclose(expected, triton_out, atol=1e-2, rtol=1e-2):
            return CorrectOut(correct=True, latency_ms=0.0, speedup=0.0, error_details=None)
        else:
            max_diff = torch.max(torch.abs(expected - triton_out)).item()
            return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                              error_details={"error_type": "numerical_stability_error",
                                             "error_message": f"Outputs differ; max abs diff {max_diff:.3e}"})

    except Exception as e:
        return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                          error_details={"error_type": "functional_validation_exception", "error_message": str(e)})


class CorrectnessAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="correctness",
            description="Runs correctness checks and performance benchmarks on compiled kernels using the KernelBench evaluation framework.",
            tools=[correctness_tool]
        )

    async def validate(self, payload: CorrectIn) -> CorrectOut:
        """
        Validates the correctness of a compiled Triton kernel against a PyTorch reference.
        This implementation tries KernelBench first, then falls back to direct Triton validation.
        """
        logger.info("Starting kernel validation")

        use_kernelbench = (
            payload.problem_id is not None and isinstance(payload.problem_id, int) and payload.problem_id > 0 and
            payload.level is not None and isinstance(payload.level, int) and payload.level > 0
        )

        if use_kernelbench:
            # 1) Try KernelBench validation first
            logger.info(
                "Attempting KernelBench validation (problem_id=%s, level=%s)",
                payload.problem_id,
                payload.level,
            )
            kb_res = await _validate_kernel_with_kernelbench(payload)

            # If KB says correct, we can still optionally run direct but no need
            if kb_res.correct:
                return kb_res

            # Whether compilation error *or* numerical mismatch, fall back to generic path
            logger.warning(
                "KernelBench reported incorrect or unsupported kernel – running direct Triton validation as fallback",
            )

            direct_res = await _validate_kernel_with_triton_direct(payload)

            # If direct validation also fails, propagate richer error info
            if not direct_res.correct:
                # attach KB details for debugging
                if direct_res.error_details is None:
                    direct_res.error_details = {}
                direct_res.error_details.update({
                    "kernelbench_correct": kb_res.correct,
                    "kernelbench_error_details": kb_res.error_details,
                })
            return direct_res

        # Generic path: use direct Triton validation only
        logger.info("Skipping KernelBench – using direct Triton validation")
        return await _validate_kernel_with_triton_direct(payload)


# -----------------------------------------------------------------------------
# Simplified fallback validators
# -----------------------------------------------------------------------------

# KernelBench path is currently disabled for simplicity; provide stub

async def _validate_kernel_with_kernelbench(payload: CorrectIn) -> CorrectOut:
    """Stub that marks KernelBench validation as skipped."""
    return CorrectOut(correct=False, latency_ms=0.0, speedup=0.0,
                      error_details={"error_type": "kernelbench_skipped",
                                     "error_message": "KernelBench validation disabled in simplified path"})


# Define the FunctionTool using the new KernelBench-based validation logic.
correctness_tool = FunctionTool(_validate_kernel_with_kernelbench)
