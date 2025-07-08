import torch
import triton
import triton.language as tl
import tempfile
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any
from utils.logging_utils import get_logger
import numpy as np

logger = get_logger("NumericalValidator")

class NumericalStabilityResult:
    """Results from numerical stability validation."""
    def __init__(self, 
                 is_stable: bool, 
                 max_error: float, 
                 mean_error: float,
                 stability_score: float,
                 issues: List[str],
                 recommendations: List[str]):
        self.is_stable = is_stable
        self.max_error = max_error
        self.mean_error = mean_error
        self.stability_score = stability_score  # 0-100, higher is better
        self.issues = issues
        self.recommendations = recommendations

class NumericalValidator:
    """Validates kernel implementations for numerical stability during generation."""
    
    def __init__(self):
        self.stability_thresholds = {
            'softmax': {'max_error': 1e-5, 'mean_error': 1e-6},
            'matmul': {'max_error': 1e-4, 'mean_error': 1e-5}, 
            'elementwise': {'max_error': 1e-6, 'mean_error': 1e-7}
        }
        
    def validate_kernel_stability(self, 
                                  kernel_src: str, 
                                  operation_type: str,
                                  input_shapes: List[List[int]],
                                  input_dtypes: List[torch.dtype],
                                  pytorch_reference: str) -> NumericalStabilityResult:
        """
        Validate a kernel for numerical stability by running it against PyTorch reference.
        This happens DURING generation, not after.
        """
        logger.info(f"Validating numerical stability for {operation_type} kernel")
        
        try:
            # Run quick stability tests
            stability_result = self._run_stability_tests(
                kernel_src, operation_type, input_shapes, input_dtypes, pytorch_reference
            )
            
            # Analyze code for known stability issues
            code_issues = self._analyze_code_stability(kernel_src, operation_type)
            stability_result.issues.extend(code_issues)
            
            # Generate recommendations
            recommendations = self._generate_stability_recommendations(
                operation_type, stability_result.issues, stability_result.max_error
            )
            stability_result.recommendations.extend(recommendations)
            
            return stability_result
            
        except Exception as e:
            logger.error(f"Stability validation failed: {e}")
            return NumericalStabilityResult(
                is_stable=False, 
                max_error=float('inf'), 
                mean_error=float('inf'),
                stability_score=0.0,
                issues=[f"Validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"]
            )
    
    def _run_stability_tests(self, 
                             kernel_src: str, 
                             operation_type: str,
                             input_shapes: List[List[int]],
                             input_dtypes: List[torch.dtype],
                             pytorch_reference: str) -> NumericalStabilityResult:
        """Run actual numerical tests against PyTorch reference."""
        
        # Create test inputs - focus on edge cases that expose numerical issues
        test_cases = self._generate_test_cases(input_shapes, input_dtypes, operation_type)
        
        max_error = 0.0
        mean_errors = []
        issues = []
        
        for i, inputs in enumerate(test_cases):
            try:
                # Get PyTorch reference result
                ref_output = self._run_pytorch_reference(pytorch_reference, inputs)
                
                # Get kernel result
                kernel_output = self._run_kernel(kernel_src, inputs)
                
                # Compare results
                error_metrics = self._compute_error_metrics(ref_output, kernel_output)
                max_error = max(max_error, error_metrics['max_error'])
                mean_errors.append(error_metrics['mean_error'])
                
                # Check for specific numerical issues
                if error_metrics['has_nan']:
                    issues.append(f"Test case {i}: NaN values detected")
                if error_metrics['has_inf']:
                    issues.append(f"Test case {i}: Infinite values detected")
                if error_metrics['max_error'] > self.stability_thresholds[operation_type]['max_error']:
                    issues.append(f"Test case {i}: High error {error_metrics['max_error']:.2e}")
                    
            except Exception as e:
                issues.append(f"Test case {i} failed: {str(e)}")
                max_error = float('inf')
        
        mean_error = np.mean(mean_errors) if mean_errors else float('inf')
        
        # Calculate stability score (0-100)
        stability_score = self._calculate_stability_score(max_error, mean_error, operation_type, issues)
        
        # Determine if stable
        thresholds = self.stability_thresholds[operation_type]
        is_stable = (max_error <= thresholds['max_error'] and 
                    mean_error <= thresholds['mean_error'] and 
                    len([issue for issue in issues if 'NaN' in issue or 'Infinite' in issue]) == 0)
        
        return NumericalStabilityResult(
            is_stable=is_stable,
            max_error=max_error,
            mean_error=mean_error,
            stability_score=stability_score,
            issues=issues,
            recommendations=[]
        )
    
    def _generate_test_cases(self, 
                             input_shapes: List[List[int]], 
                             input_dtypes: List[torch.dtype],
                             operation_type: str) -> List[List[torch.Tensor]]:
        """Generate test cases that expose numerical instability."""
        
        test_cases = []
        
        # Standard random case
        case1 = []
        for shape, dtype in zip(input_shapes, input_dtypes):
            case1.append(torch.randn(shape, dtype=dtype, device='cuda'))
        test_cases.append(case1)
        
        # Large values case (tests overflow)
        case2 = []
        for shape, dtype in zip(input_shapes, input_dtypes):
            tensor = torch.randn(shape, dtype=dtype, device='cuda') * 10
            case2.append(tensor)
        test_cases.append(case2)
        
        # Small values case (tests underflow)
        case3 = []
        for shape, dtype in zip(input_shapes, input_dtypes):
            tensor = torch.randn(shape, dtype=dtype, device='cuda') * 0.01
            case3.append(tensor)
        test_cases.append(case3)
        
        # Operation-specific edge cases
        if operation_type == 'softmax':
            # Very large values that should trigger softmax overflow without proper max subtraction
            case4 = []
            for shape, dtype in zip(input_shapes, input_dtypes):
                tensor = torch.full(shape, 100.0, dtype=dtype, device='cuda')
                tensor += torch.randn_like(tensor) * 0.1  # Add small noise
                case4.append(tensor)
            test_cases.append(case4)
            
            # Mixed very large and very small values
            case5 = []
            for shape, dtype in zip(input_shapes, input_dtypes):
                tensor = torch.randn(shape, dtype=dtype, device='cuda')
                tensor[..., :shape[-1]//2] *= 100  # Large values
                tensor[..., shape[-1]//2:] *= 0.01  # Small values
                case5.append(tensor)
            test_cases.append(case5)
        
        return test_cases
    
    def _run_pytorch_reference(self, pytorch_reference: str, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Execute PyTorch reference implementation."""
        # Create a temporary namespace for execution
        namespace = {
            'torch': torch,
            'input0': inputs[0] if len(inputs) > 0 else None,
            'input1': inputs[1] if len(inputs) > 1 else None,
            'input2': inputs[2] if len(inputs) > 2 else None,
        }
        
        # Extract the return statement from pytorch_reference
        lines = pytorch_reference.strip().split('\n')
        for line in lines:
            if 'return ' in line:
                return_expr = line.split('return ')[1].strip()
                result = eval(return_expr, namespace)
                return result
        
        raise ValueError("No return statement found in PyTorch reference")
    
    def _run_kernel(self, kernel_src: str, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Execute the kernel implementation."""
        # Write kernel to temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(kernel_src)
            f.flush()
            
            # Import and execute the kernel
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_kernel", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the launcher function
            launcher_func = None
            for attr_name in dir(module):
                if attr_name.startswith('launch_'):
                    launcher_func = getattr(module, attr_name)
                    break
            
            if not launcher_func:
                raise ValueError("No launcher function found in kernel")
            
            # Create output tensor
            output_shape = inputs[0].shape  # Assume same shape as first input for now
            output = torch.empty(output_shape, dtype=inputs[0].dtype, device='cuda')
            
            # Launch kernel
            launcher_func(output, *inputs)
            
            return output
    
    def _compute_error_metrics(self, reference: torch.Tensor, kernel_output: torch.Tensor) -> Dict[str, Any]:
        """Compute comprehensive error metrics."""
        
        # Convert to CPU for analysis
        ref_cpu = reference.cpu().float()
        kernel_cpu = kernel_output.cpu().float()
        
        # Absolute difference
        abs_diff = torch.abs(ref_cpu - kernel_cpu)
        
        # Relative difference (avoid division by zero)
        rel_diff = abs_diff / (torch.abs(ref_cpu) + 1e-12)
        
        return {
            'max_error': float(torch.max(abs_diff)),
            'mean_error': float(torch.mean(abs_diff)),
            'max_rel_error': float(torch.max(rel_diff)),
            'mean_rel_error': float(torch.mean(rel_diff)),
            'has_nan': bool(torch.isnan(kernel_cpu).any()),
            'has_inf': bool(torch.isinf(kernel_cpu).any()),
            'mse': float(torch.mean((ref_cpu - kernel_cpu) ** 2))
        }
    
    def _analyze_code_stability(self, kernel_src: str, operation_type: str) -> List[str]:
        """Analyze kernel source code for known numerical stability issues."""
        issues = []
        
        if operation_type == 'softmax':
            # Check for common softmax stability issues
            if 'tl.exp(' in kernel_src and 'tl.max(' not in kernel_src:
                issues.append("Missing max subtraction for numerical stability")
            
            if '.to(tl.float32)' not in kernel_src:
                issues.append("Missing FP32 accumulation for precision")
            
            if 'denominator' in kernel_src and '+' not in kernel_src.split('denominator')[1].split('\n')[0]:
                if 'epsilon' not in kernel_src.lower():
                    issues.append("Missing epsilon protection against division by zero")
            
            # Check for proper block size handling
            if 'BLOCK_SIZE' in kernel_src:
                if '16384' in kernel_src or '8192' in kernel_src:
                    # Check if configs support these sizes
                    if '16384' not in kernel_src or '8192' not in kernel_src:
                        issues.append("BLOCK_SIZE configs may not support large sequence lengths")
        
        return issues
    
    def _calculate_stability_score(self, 
                                   max_error: float, 
                                   mean_error: float, 
                                   operation_type: str,
                                   issues: List[str]) -> float:
        """Calculate a stability score from 0-100."""
        
        if max_error == float('inf') or mean_error == float('inf'):
            return 0.0
        
        thresholds = self.stability_thresholds[operation_type]
        
        # Error score (0-50 points)
        max_error_score = max(0, 25 - 25 * (max_error / thresholds['max_error']))
        mean_error_score = max(0, 25 - 25 * (mean_error / thresholds['mean_error']))
        
        # Issue penalty (0-50 points)
        critical_issues = len([issue for issue in issues if any(word in issue.lower() 
                                                               for word in ['nan', 'inf', 'overflow', 'underflow'])])
        other_issues = len(issues) - critical_issues
        
        issue_penalty = critical_issues * 20 + other_issues * 5
        issue_score = max(0, 50 - issue_penalty)
        
        total_score = max_error_score + mean_error_score + issue_score
        return min(100.0, total_score)
    
    def _generate_stability_recommendations(self, 
                                            operation_type: str, 
                                            issues: List[str], 
                                            max_error: float) -> List[str]:
        """Generate specific recommendations for improving stability."""
        recommendations = []
        
        if operation_type == 'softmax':
            if any('max subtraction' in issue for issue in issues):
                recommendations.append("Add row_max = tl.max(row, axis=0); row = row - row_max before tl.exp()")
            
            if any('FP32 accumulation' in issue for issue in issues):
                recommendations.append("Convert intermediate values to FP32: .to(tl.float32)")
            
            if any('epsilon protection' in issue for issue in issues):
                recommendations.append("Add small epsilon to denominator: denominator + 1e-8")
            
            if any('BLOCK_SIZE' in issue for issue in issues):
                recommendations.append("Add larger BLOCK_SIZE configs (8192, 16384) to autotune")
        
        if max_error > 1e-3:
            recommendations.append("Consider using higher precision arithmetic throughout")
        
        return recommendations 