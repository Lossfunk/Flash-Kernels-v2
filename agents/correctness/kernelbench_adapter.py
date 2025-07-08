"""
KernelBench Interface Adapter

This module provides automatic adaptation between generated Triton kernels and KernelBench's
expected interface. It handles different launcher function signatures and ensures compatibility.
"""

import torch
import torch.nn as nn
import ast
import inspect
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from utils.logging_utils import get_logger

logger = get_logger("KernelBenchAdapter")


class LauncherSignatureAnalyzer:
    """Analyzes launcher function signatures to determine the correct calling convention."""
    
    def __init__(self):
        self.signature_patterns = {
            'in_place_with_output': r'def\s+(\w+)\s*\(\s*output\s*,',  # launch_func(output, *inputs)
            'in_place_first_param': r'def\s+(\w+)\s*\(\s*(\w+)\s*,',   # launch_func(input_output, *other_inputs)  
            'functional_return': r'def\s+(\w+)\s*\([^)]*\)\s*->',      # launch_func(*inputs) -> output
            'functional_no_annotation': r'def\s+(\w+)\s*\([^)]*\):',   # launch_func(*inputs) (no annotation)
        }
    
    def analyze_launcher_signature(self, kernel_src: str, launcher_name: str) -> Dict[str, Any]:
        """
        Analyze the launcher function signature to determine calling convention.
        
        Returns:
            Dict with signature analysis results:
            - 'type': 'in_place_with_output', 'in_place_first_param', 'functional_return', 'functional_no_annotation'
            - 'param_names': List of parameter names
            - 'has_output_param': Whether function has explicit output parameter
            - 'return_annotation': Whether function has return type annotation
        """
        try:
            # Parse the AST to get accurate signature information
            tree = ast.parse(kernel_src)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == launcher_name:
                    param_names = [arg.arg for arg in node.args.args]
                    has_return_annotation = node.returns is not None
                    
                    # Determine signature type
                    if len(param_names) > 0:
                        first_param = param_names[0].lower()
                        
                        if first_param == 'output':
                            sig_type = 'in_place_with_output'
                            has_output_param = True
                        elif 'output' in first_param or first_param.endswith('_out'):
                            sig_type = 'in_place_first_param'  
                            has_output_param = True
                        elif has_return_annotation:
                            sig_type = 'functional_return'
                            has_output_param = False
                        else:
                            # Need to check if function actually returns something
                            returns_value = self._check_if_function_returns(node)
                            if returns_value:
                                sig_type = 'functional_return'
                                has_output_param = False
                            else:
                                sig_type = 'in_place_first_param'  # Assume first param is modified
                                has_output_param = True
                    else:
                        sig_type = 'functional_no_annotation'
                        has_output_param = False
                    
                    return {
                        'type': sig_type,
                        'param_names': param_names,
                        'has_output_param': has_output_param,
                        'return_annotation': has_return_annotation,
                        'input_param_count': len(param_names) - (1 if has_output_param else 0)
                    }
            
            # Fallback to regex if AST parsing fails
            return self._analyze_with_regex(kernel_src, launcher_name)
            
        except Exception as e:
            logger.warning(f"Failed to analyze launcher signature with AST: {e}")
            return self._analyze_with_regex(kernel_src, launcher_name)
    
    def _check_if_function_returns(self, func_node: ast.FunctionDef) -> bool:
        """Check if a function has return statements that return values."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False
    
    def _analyze_with_regex(self, kernel_src: str, launcher_name: str) -> Dict[str, Any]:
        """Fallback regex-based signature analysis."""
        lines = kernel_src.split('\n')
        
        for line in lines:
            if f'def {launcher_name}(' in line:
                # Extract parameter list
                match = re.search(rf'def\s+{re.escape(launcher_name)}\s*\(([^)]*)\)', line)
                if match:
                    params_str = match.group(1).strip()
                    if params_str:
                        param_names = [p.strip().split(':')[0].split('=')[0].strip() 
                                     for p in params_str.split(',') if p.strip()]
                    else:
                        param_names = []
                    
                    # Determine type based on first parameter
                    if param_names and param_names[0].lower() == 'output':
                        return {
                            'type': 'in_place_with_output',
                            'param_names': param_names,
                            'has_output_param': True,
                            'return_annotation': '->' in line,
                            'input_param_count': len(param_names) - 1
                        }
                    else:
                        return {
                            'type': 'functional_no_annotation',
                            'param_names': param_names,
                            'has_output_param': False,
                            'return_annotation': '->' in line,
                            'input_param_count': len(param_names)
                        }
        
        # Default fallback
        return {
            'type': 'functional_no_annotation',
            'param_names': [],
            'has_output_param': False,
            'return_annotation': False,
            'input_param_count': 0
        }


class KernelBenchModelAdapter:
    """Adapts generated Triton kernels to KernelBench's expected interface."""
    
    def __init__(self):
        self.analyzer = LauncherSignatureAnalyzer()
    
    def create_adapted_model_class(self, kernel_src: str, input_specs: List[Any], 
                                 launcher_name: str) -> str:
        """
        Create a KernelBench-compatible ModelNew class that adapts the launcher function.
        
        Args:
            kernel_src: The generated Triton kernel source code
            input_specs: List of input tensor specifications
            launcher_name: Name of the launcher function to adapt
            
        Returns:
            Complete Python source code for the adapted ModelNew class
        """
        logger.info(f"Creating adapted ModelNew class for launcher: {launcher_name}")
        
        # Analyze the launcher signature
        sig_analysis = self.analyzer.analyze_launcher_signature(kernel_src, launcher_name)
        logger.info(f"Launcher signature analysis: {sig_analysis}")
        
        # Generate forward method parameters
        forward_params = []
        forward_args = []
        
        for i, spec in enumerate(input_specs):
            param_name = chr(ord('A') + i) if i < 26 else f"input_{i}"
            forward_params.append(f"{param_name}: torch.Tensor")
            forward_args.append(param_name)
        
        forward_signature = ", ".join(forward_params)
        forward_call_args = ", ".join(forward_args)
        
        # Generate output tensor creation logic
        output_creation = self._generate_output_creation(input_specs, forward_args)
        
        # Generate the appropriate kernel call based on signature type
        kernel_call = self._generate_kernel_call(
            launcher_name, sig_analysis, forward_args, output_creation
        )
        
        # Build the complete adapted model
        adapted_model = f'''import torch
import torch.nn as nn

# Original kernel source
{kernel_src}

class ModelNew(nn.Module):
    """KernelBench-compatible adapter for the generated Triton kernel."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, {forward_signature}) -> torch.Tensor:
        """
        Forward pass that adapts the launcher function to KernelBench's expected interface.
        
        Launcher signature type: {sig_analysis['type']}
        Original launcher: {launcher_name}({', '.join(sig_analysis['param_names'])})
        """
        {kernel_call}
'''
        
        logger.debug(f"Generated adapted ModelNew class:\\n{adapted_model}")
        return adapted_model
    
    def _generate_output_creation(self, input_specs: List[Any], forward_args: List[str]) -> str:
        """Generate appropriate output tensor creation code."""
        if len(input_specs) >= 2 and hasattr(input_specs[0], 'shape') and hasattr(input_specs[1], 'shape'):
            # Check if this looks like matrix multiplication
            if (len(input_specs[0].shape) == 2 and len(input_specs[1].shape) == 2 and 
                len(input_specs) == 2):
                return f"output = torch.empty(({forward_args[0]}.shape[0], {forward_args[1]}.shape[1]), device={forward_args[0]}.device, dtype={forward_args[0]}.dtype)"
        
        # Default: same shape as first input
        return f"output = torch.empty_like({forward_args[0]})"
    
    def _generate_kernel_call(self, launcher_name: str, sig_analysis: Dict[str, Any], 
                            forward_args: List[str], output_creation: str) -> str:
        """Generate the appropriate kernel call based on signature analysis."""
        
        if sig_analysis['type'] == 'in_place_with_output':
            # launcher(output, *inputs)
            return f"""        {output_creation}
        {launcher_name}(output, {', '.join(forward_args)})
        return output"""
        
        elif sig_analysis['type'] == 'in_place_first_param':
            # launcher(input_output, *other_inputs) - first input is modified in place
            if len(forward_args) == 1:
                return f"""        output = {forward_args[0]}.clone()
        {launcher_name}(output)
        return output"""
            else:
                return f"""        output = {forward_args[0]}.clone()
        {launcher_name}(output, {', '.join(forward_args[1:])})
        return output"""
        
        elif sig_analysis['type'] in ['functional_return', 'functional_no_annotation']:
            # launcher(*inputs) -> output or launcher(*inputs) (assuming return)
            return f"""        return {launcher_name}({', '.join(forward_args)})"""
        
        else:
            # Fallback: try in_place_with_output pattern
            logger.warning(f"Unknown signature type {sig_analysis['type']}, using fallback")
            return f"""        {output_creation}
        {launcher_name}(output, {', '.join(forward_args)})
        return output"""


def adapt_kernel_for_kernelbench(kernel_src: str, input_specs: List[Any]) -> str:
    """
    Main entry point for adapting a Triton kernel to KernelBench interface.
    
    Args:
        kernel_src: Generated Triton kernel source code
        input_specs: List of input tensor specifications
        
    Returns:
        Complete adapted Python source code ready for KernelBench
    """
    adapter = KernelBenchModelAdapter()
    
    # Find the launcher function
    launcher_name = None
    lines = kernel_src.split('\n')
    
    # First, try to find a launch_ function (preferred)
    for line in lines:
        if line.strip().startswith('def launch_'):
            launcher_name = line.strip().split('(')[0].replace('def ', '')
            logger.info(f"Found preferred launch_ function: {launcher_name}")
            break
    
    # If no launch_ function found, look for any function that calls a kernel
    if not launcher_name:
        try:
            tree = ast.parse(kernel_src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this function contains kernel calls (kernel[grid](...))
                    for child in ast.walk(node):
                        if isinstance(child, ast.Subscript):
                            # Look for pattern like kernel_name[grid](...)
                            if isinstance(child.value, ast.Name):
                                launcher_name = node.name
                                logger.info(f"Found kernel-calling function: {launcher_name}")
                                break
                    if launcher_name:
                        break
        except Exception as e:
            logger.warning(f"Failed to parse kernel source for function detection: {e}")
    
    if not launcher_name:
        raise ValueError("Could not find launcher function in Triton kernel source")
    
    return adapter.create_adapted_model_class(kernel_src, input_specs, launcher_name) 