"""
Operation-level correctness testing inspired by Liger's approach.
Tests individual operations rather than trying to build reference models.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable, Any, Optional
import pytest
from pathlib import Path
import importlib.util
import traceback

from agents.contracts import TensorSpec, CorrectOut
from utils.logging_utils import get_logger

logger = get_logger("OperationTests")


class OperationTester:
    """Test individual operations against PyTorch reference implementations."""
    
    def __init__(self):
        self.test_cases = {
            'matmul': self._test_matmul,
            'softmax': self._test_softmax,
            'layer_norm': self._test_layer_norm,
            'relu': self._test_relu,
            'linear': self._test_linear,
            'mlp': self._test_mlp,
        }
    
    def test_operation(self, 
                      triton_kernel_path: str,
                      operation_type: str,
                      input_specs: List[TensorSpec],
                      **kwargs) -> CorrectOut:
        """Test a specific operation type against PyTorch reference."""
        
        if operation_type not in self.test_cases:
            return CorrectOut(
                correct=False,
                latency_ms=0.0,
                speedup=0.0,
                error_details={
                    "error_type": "unsupported_operation",
                    "error_message": f"Operation type '{operation_type}' not supported. Available: {list(self.test_cases.keys())}"
                }
            )
        
        try:
            # Load the Triton kernel with enhanced error handling
            triton_func = self._load_triton_kernel_safe(triton_kernel_path)
            
            # Generate test inputs with validation
            test_inputs = self._generate_test_inputs_safe(input_specs)
            
            # Run the specific test with comprehensive error handling
            test_func = self.test_cases[operation_type]
            return test_func(triton_func, test_inputs, **kwargs)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Operation test failed for {operation_type}: {str(e)}")
            logger.debug(f"Full traceback: {error_trace}")
            
            return CorrectOut(
                correct=False,
                latency_ms=0.0,
                speedup=0.0,
                error_details={
                    "error_type": "test_execution_error",
                    "error_message": str(e),
                    "full_traceback": error_trace
                }
            )
    
    def _load_triton_kernel_safe(self, kernel_path: str) -> Callable:
        """Load Triton kernel with comprehensive error handling and validation."""
        try:
            spec = importlib.util.spec_from_file_location("triton_kernel", kernel_path)
            if spec is None:
                raise RuntimeError(f"Could not load spec from {kernel_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Strategy 1: Look for launcher functions (high-level functions that call kernels)
            launcher_candidates = []
            kernel_candidates = []
            
            for name in dir(module):
                if name.startswith('_'):
                    continue
                    
                obj = getattr(module, name)
                if not callable(obj):
                    continue
                
                # Categorize functions
                if hasattr(obj, '__wrapped__') and hasattr(obj, 'cache'):
                    kernel_candidates.append((name, obj))
                elif 'triton.runtime.jit.JITFunction' in str(type(obj)):
                    kernel_candidates.append((name, obj))
                else:
                    launcher_candidates.append((name, obj))
            
            logger.info(f"Found {len(launcher_candidates)} launcher candidates: {[name for name, _ in launcher_candidates]}")
            logger.info(f"Found {len(kernel_candidates)} kernel candidates: {[name for name, _ in kernel_candidates]}")
            
            # Strategy 2: Prefer specific naming patterns for launchers
            for name, obj in launcher_candidates:
                if name.startswith('launch_'):
                    logger.info(f"Using launcher function: {name}")
                    return obj
            
            # Strategy 3: Look for operation-specific launcher names
            for name, obj in launcher_candidates:
                if name in ['softmax', 'matmul', 'layer_norm', 'relu', 'linear']:
                    logger.info(f"Using operation-specific launcher: {name}")
                    return obj
            
            # Strategy 4: Look for functions that don't end with '_kernel' 
            for name, obj in launcher_candidates:
                if not name.endswith('_kernel'):
                    logger.info(f"Using non-kernel function: {name}")
                    return obj
            
            # Strategy 5: First launcher candidate
            if launcher_candidates:
                name, obj = launcher_candidates[0]
                logger.info(f"Using first launcher candidate: {name}")
                return obj
            
            # Strategy 6: If no launchers, try to create a wrapper for the kernel
            if kernel_candidates:
                name, kernel_func = kernel_candidates[0]
                logger.warning(f"No launcher found, attempting to wrap kernel: {name}")
                return self._create_kernel_wrapper(kernel_func, name)
            
            raise RuntimeError(f"No suitable function found in {kernel_path}. Available: {[name for name in dir(module) if callable(getattr(module, name))]}")
            
        except Exception as e:
            logger.error(f"Failed to load kernel from {kernel_path}: {str(e)}")
            raise RuntimeError(f"Kernel loading failed: {str(e)}")
    
    def _create_kernel_wrapper(self, kernel_func: Callable, kernel_name: str) -> Callable:
        """Create a wrapper for a raw Triton kernel when no launcher is available."""
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"Auto-wrapper for kernel {kernel_name} not implemented. Please provide a launcher function.")
        return wrapper
    
    def _generate_test_inputs_safe(self, input_specs: List[TensorSpec]) -> List[torch.Tensor]:
        """Generate test inputs with enhanced validation."""
        if not input_specs:
            raise ValueError("No input specifications provided")
        
        inputs = []
        for i, spec in enumerate(input_specs):
            try:
                shape = tuple(spec.shape) if spec.shape else ()
                if not shape:
                    raise ValueError(f"Input spec {i} has empty shape")
                
                if spec.dtype == "float16":
                    tensor = torch.randn(*shape, dtype=torch.float16, device="cuda")
                elif spec.dtype == "int32":
                    tensor = torch.randint(0, 10, shape, dtype=torch.int32, device="cuda")
                elif spec.dtype == "int8":
                    tensor = torch.randint(-128, 127, shape, dtype=torch.int8, device="cuda")
                else:  # default float32
                    tensor = torch.randn(*shape, dtype=torch.float32, device="cuda")
                
                # Validate tensor was created successfully
                if tensor.numel() == 0:
                    raise ValueError(f"Generated tensor {i} has zero elements")
                
                inputs.append(tensor)
                logger.debug(f"Generated input {i}: shape={tensor.shape}, dtype={tensor.dtype}")
                
            except Exception as e:
                raise ValueError(f"Failed to generate input {i} from spec {spec}: {str(e)}")
        
        return inputs
    
    def _test_matmul(self, triton_func: Callable, inputs: List[torch.Tensor], **kwargs) -> CorrectOut:
        """Test matrix multiplication operation."""
        if len(inputs) < 2:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "insufficient_inputs", "error_message": "MatMul requires at least 2 inputs"}
            )
        
        a, b = inputs[0], inputs[1]
        
        # PyTorch reference
        torch_result = torch.matmul(a, b)
        
        # Triton implementation
        try:
            # Try direct call first
            triton_result = triton_func(a, b)
        except TypeError:
            # Try with output tensor
            output = torch.empty_like(torch_result)
            triton_func(output, a, b)
            triton_result = output
        
        return self._compare_outputs(torch_result, triton_result, "matmul")
    
    def _test_softmax(self, triton_func: Callable, inputs: List[torch.Tensor], **kwargs) -> CorrectOut:
        """Test softmax operation."""
        if len(inputs) < 1:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "insufficient_inputs", "error_message": "Softmax requires at least 1 input"}
            )
        
        x = inputs[0]
        dim = kwargs.get('dim', -1)
        
        # PyTorch reference
        torch_result = F.softmax(x, dim=dim)
        
        # Triton implementation
        try:
            triton_result = triton_func(x, dim=dim)
        except TypeError:
            try:
                triton_result = triton_func(x)
            except TypeError:
                output = torch.empty_like(x)
                triton_func(output, x)
                triton_result = output
        
        return self._compare_outputs(torch_result, triton_result, "softmax")
    
    def _test_layer_norm(self, triton_func: Callable, inputs: List[torch.Tensor], **kwargs) -> CorrectOut:
        """Test layer normalization operation with comprehensive error handling."""
        if len(inputs) < 1:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "insufficient_inputs", "error_message": "LayerNorm requires at least 1 input"}
            )
        
        x = inputs[0]
        logger.info(f"Testing LayerNorm with input shape: {x.shape}")
        
        # Handle different input shapes safely
        if len(x.shape) < 1:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "invalid_input_shape", "error_message": f"Input must have at least 1 dimension, got shape {x.shape}"}
            )
        
        # Determine normalized shape - default to last dimension
        normalized_shape = kwargs.get('normalized_shape', (x.shape[-1],))
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        eps = kwargs.get('eps', 1e-5)
        
        logger.info(f"LayerNorm normalized_shape: {normalized_shape}, eps: {eps}")
        
        # Create weight and bias for fair comparison
        try:
            weight = torch.ones(normalized_shape, device=x.device, dtype=x.dtype)
            bias = torch.zeros(normalized_shape, device=x.device, dtype=x.dtype)
        except Exception as e:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "weight_bias_creation_error", "error_message": f"Failed to create weight/bias tensors: {str(e)}"}
            )
        
        # PyTorch reference
        try:
            torch_result = F.layer_norm(x, normalized_shape, weight, bias, eps)
            logger.info(f"PyTorch LayerNorm result shape: {torch_result.shape}")
        except Exception as e:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "pytorch_reference_error", "error_message": f"PyTorch LayerNorm failed: {str(e)}"}
            )
        
        # Triton implementation with multiple fallback strategies
        triton_result = None
        last_error = None
        
        # Strategy 1: Direct call with all parameters
        try:
            triton_result = triton_func(x, weight, bias, eps)
            logger.info("LayerNorm Strategy 1 (direct call) succeeded")
        except Exception as e:
            last_error = f"Strategy 1 failed: {str(e)}"
            logger.debug(last_error)
        
        # Strategy 2: With output tensor
        if triton_result is None:
            try:
                output = torch.empty_like(torch_result)
                triton_func(output, x, weight, bias, eps)
                triton_result = output
                logger.info("LayerNorm Strategy 2 (with output tensor) succeeded")
            except Exception as e:
                last_error = f"Strategy 2 failed: {str(e)}"
                logger.debug(last_error)
        
        # Strategy 3: Only input (if weight/bias are hardcoded)
        if triton_result is None:
            try:
                triton_result = triton_func(x)
                logger.info("LayerNorm Strategy 3 (input only) succeeded")
            except Exception as e:
                last_error = f"Strategy 3 failed: {str(e)}"
                logger.debug(last_error)
        
        # Strategy 4: Input + output only
        if triton_result is None:
            try:
                output = torch.empty_like(torch_result)
                triton_func(output, x)
                triton_result = output
                logger.info("LayerNorm Strategy 4 (input + output) succeeded")
            except Exception as e:
                last_error = f"Strategy 4 failed: {str(e)}"
                logger.debug(last_error)
        
        if triton_result is None:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={
                    "error_type": "triton_execution_error", 
                    "error_message": f"All LayerNorm execution strategies failed. Last error: {last_error}"
                }
            )
        
        return self._compare_outputs(torch_result, triton_result, "layer_norm")
    
    def _test_relu(self, triton_func: Callable, inputs: List[torch.Tensor], **kwargs) -> CorrectOut:
        """Test ReLU activation operation."""
        if len(inputs) < 1:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "insufficient_inputs", "error_message": "ReLU requires at least 1 input"}
            )
        
        x = inputs[0]
        
        # PyTorch reference
        torch_result = F.relu(x)
        
        # Triton implementation
        try:
            triton_result = triton_func(x)
        except TypeError:
            output = torch.empty_like(x)
            triton_func(output, x)
            triton_result = output
        
        return self._compare_outputs(torch_result, triton_result, "relu")
    
    def _test_linear(self, triton_func: Callable, inputs: List[torch.Tensor], **kwargs) -> CorrectOut:
        """Test linear (fully connected) operation."""
        if len(inputs) < 2:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "insufficient_inputs", "error_message": "Linear requires at least 2 inputs (input, weight)"}
            )
        
        x, weight = inputs[0], inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        
        # PyTorch reference
        torch_result = F.linear(x, weight, bias)
        
        # Triton implementation
        try:
            if bias is not None:
                triton_result = triton_func(x, weight, bias)
            else:
                triton_result = triton_func(x, weight)
        except TypeError:
            output = torch.empty_like(torch_result)
            if bias is not None:
                triton_func(output, x, weight, bias)
            else:
                triton_func(output, x, weight)
            triton_result = output
        
        return self._compare_outputs(torch_result, triton_result, "linear")
    
    def _test_mlp(self, triton_func: Callable, inputs: List[torch.Tensor], **kwargs) -> CorrectOut:
        """Test MLP (multi-layer perceptron) operation."""
        if len(inputs) < 1:
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={"error_type": "insufficient_inputs", "error_message": "MLP requires at least 1 input"}
            )
        
        x = inputs[0]
        dim = kwargs.get('dim', x.shape[-1])
        out_dim = kwargs.get('out', dim)
        
        # Create MLP weights for fair comparison (matching the model structure)
        device, dtype = x.device, x.dtype
        fc1_weight = torch.randn(dim, dim, device=device, dtype=dtype) * 0.1
        fc1_bias = torch.zeros(dim, device=device, dtype=dtype)
        fc2_weight = torch.randn(out_dim, dim, device=device, dtype=dtype) * 0.1
        fc2_bias = torch.zeros(out_dim, device=device, dtype=dtype)
        
        # PyTorch reference MLP
        h1 = F.linear(x, fc1_weight, fc1_bias)
        h1_relu = F.relu(h1)
        torch_result = F.linear(h1_relu, fc2_weight, fc2_bias)
        
        # Triton implementation - try different calling patterns
        try:
            # Pattern 1: Direct call with all parameters
            triton_result = triton_func(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        except TypeError:
            try:
                # Pattern 2: With output tensor
                output = torch.empty_like(torch_result)
                triton_func(output, x, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
                triton_result = output
            except TypeError:
                try:
                    # Pattern 3: Just input (if weights are hardcoded)
                    triton_result = triton_func(x)
                except TypeError:
                    # Pattern 4: Input + output
                    output = torch.empty_like(torch_result)
                    triton_func(output, x)
                    triton_result = output
        
        return self._compare_outputs(torch_result, triton_result, "mlp")
    
    def _compare_outputs(self, torch_result: torch.Tensor, triton_result: torch.Tensor, op_name: str) -> CorrectOut:
        """Compare PyTorch and Triton outputs with comprehensive error handling."""
        try:
            # Shape check
            if torch_result.shape != triton_result.shape:
                return CorrectOut(
                    correct=False, latency_ms=0.0, speedup=0.0,
                    error_details={
                        "error_type": "shape_mismatch",
                        "error_message": f"{op_name}: Expected shape {torch_result.shape}, got {triton_result.shape}",
                        "expected_shape": list(torch_result.shape),
                        "actual_shape": list(triton_result.shape)
                    }
                )
            
            # Check for NaN or Inf values
            if torch.isnan(torch_result).any() or torch.isinf(torch_result).any():
                return CorrectOut(
                    correct=False, latency_ms=0.0, speedup=0.0,
                    error_details={
                        "error_type": "invalid_pytorch_result",
                        "error_message": f"{op_name}: PyTorch result contains NaN or Inf values"
                    }
                )
            
            if torch.isnan(triton_result).any() or torch.isinf(triton_result).any():
                return CorrectOut(
                    correct=False, latency_ms=0.0, speedup=0.0,
                    error_details={
                        "error_type": "invalid_triton_result",
                        "error_message": f"{op_name}: Triton result contains NaN or Inf values"
                    }
                )
            
            # Numerical comparison with operation-specific tolerances
            tolerances = {
                'matmul': (1e-3, 1e-3),
                'softmax': (1e-4, 1e-4),  # Softmax needs tighter tolerance
                'layer_norm': (1e-4, 1e-4),
                'relu': (1e-6, 1e-6),  # ReLU should be exact
                'linear': (1e-3, 1e-3),
                'mlp': (1e-2, 1e-2),  # MLP is composite, needs looser tolerance
            }
            
            atol, rtol = tolerances.get(op_name, (1e-3, 1e-3))
            
            if torch.allclose(torch_result, triton_result, atol=atol, rtol=rtol):
                logger.info(f"{op_name} correctness test PASSED")
                return CorrectOut(correct=True, latency_ms=0.0, speedup=0.0, error_details=None)
            else:
                max_diff = torch.max(torch.abs(torch_result - triton_result)).item()
                mean_diff = torch.mean(torch.abs(torch_result - triton_result)).item()
                
                logger.warning(f"{op_name} correctness test FAILED: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
                
                return CorrectOut(
                    correct=False, latency_ms=0.0, speedup=0.0,
                    error_details={
                        "error_type": "numerical_mismatch",
                        "error_message": f"{op_name}: Max diff {max_diff:.2e}, Mean diff {mean_diff:.2e}, Tolerance atol={atol}, rtol={rtol}",
                        "max_difference": max_diff,
                        "mean_difference": mean_diff,
                        "tolerance_atol": atol,
                        "tolerance_rtol": rtol
                    }
                )
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Output comparison failed for {op_name}: {str(e)}")
            return CorrectOut(
                correct=False, latency_ms=0.0, speedup=0.0,
                error_details={
                    "error_type": "comparison_error",
                    "error_message": f"Failed to compare {op_name} outputs: {str(e)}",
                    "full_traceback": error_trace
                }
            )


# Global instance for easy access
operation_tester = OperationTester()


def test_triton_operation(kernel_path: str, 
                         operation_type: str, 
                         input_specs: List[TensorSpec], 
                         **kwargs) -> CorrectOut:
    """Convenience function for testing operations."""
    return operation_tester.test_operation(kernel_path, operation_type, input_specs, **kwargs) 