"""
Triton Kernel Validator

This module provides comprehensive validation of Triton kernels including:
- Syntax validation
- Semantic analysis
- Triton API compliance
- Signature validation
- Memory access pattern analysis
- Critical error pattern detection
"""

import ast
import re
import inspect
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from agents.contracts import TensorSpec
from utils.logging_utils import get_logger

logger = get_logger("KernelValidator")


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in kernel code."""
    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of kernel validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    fixed_code: Optional[str] = None  # Contains auto-fixed code if fixes were applied
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def get_error_summary(self) -> str:
        """Get a summary of all errors."""
        errors = [issue.message for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
        return "; ".join(errors) if errors else "No errors"


class TritonErrorPatternValidator:
    """Validates against known Triton error patterns that cause compilation failures."""
    
    def validate_error_patterns(self, code: str) -> List[ValidationIssue]:
        """Validate against known error patterns."""
        issues = []
        
        # Check for the specific tl.max axis=0 error pattern
        issues.extend(self._validate_reduction_axis_errors(code))
        
        # Check for incorrect tl.load patterns
        issues.extend(self._validate_load_patterns(code))
        
        # Check for launcher function errors
        issues.extend(self._validate_launcher_patterns(code))
        
        # Check for deprecated API usage
        issues.extend(self._validate_deprecated_apis(code))
        
        return issues
    
    def _validate_reduction_axis_errors(self, code: str) -> List[ValidationIssue]:
        """Detect patterns that cause 'invalid axis' errors with reduction operations."""
        issues = []
        
        # Pattern 1: tl.load creating scalar then tl.max with axis=0
        # This is the exact error from the logs: row_data = tl.load(row_start_ptr, N); tl.max(row_data, axis=0)
        scalar_load_pattern = r'(\w+)\s*=\s*tl\.load\([^,]+,\s*\w+\s*\)'
        max_with_axis_pattern = r'tl\.max\(\s*(\w+)\s*,\s*axis\s*=\s*0\s*\)'
        
        scalar_loads = re.finditer(scalar_load_pattern, code)
        scalar_vars = set()
        
        for match in scalar_loads:
            var_name = match.group(1)
            scalar_vars.add(var_name)
        
        max_calls = re.finditer(max_with_axis_pattern, code)
        for match in max_calls:
            var_name = match.group(1)
            if var_name in scalar_vars:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid axis=0 on scalar variable '{var_name}'. Variable loaded as scalar with tl.load(ptr, size) creates a scalar, not a tensor.",
                    suggestion=f"Use tl.load({var_name}_ptr + offsets, mask=mask) with proper offsets to create a tensor, or remove axis=0 for scalar operations"
                ))
        
        # Pattern 2: General axis usage on potentially scalar values
        axis_patterns = [
            (r'tl\.sum\(\s*(\w+)\s*,\s*axis\s*=\s*0\s*\)', 'tl.sum'),
            (r'tl\.mean\(\s*(\w+)\s*,\s*axis\s*=\s*0\s*\)', 'tl.mean'),
            (r'tl\.min\(\s*(\w+)\s*,\s*axis\s*=\s*0\s*\)', 'tl.min'),
        ]
        
        for pattern, func_name in axis_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                var_name = match.group(1)
                if var_name in scalar_vars:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid axis=0 on scalar variable '{var_name}' in {func_name}()",
                        suggestion=f"Ensure '{var_name}' is a tensor with proper dimensions, or remove axis parameter for scalar operations"
                    ))
        
        # NEW Pattern 3: usage of unsupported 'keepdim' keyword in Triton reductions
        # Triton (2.x/3.x) does not accept keepdim – any occurrence will lead to a TypeError at compile time.
        keepdim_pattern = r"tl\.(max|min|sum|mean|argmax|argmin|logsumexp)\([^)]*keepdim\s*=\s*[^),]+[^)]*\)"
        for _ in re.finditer(keepdim_pattern, code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Triton reduction ops (tl.max/tl.sum/…) do NOT support the 'keepdim' parameter.",
                suggestion="Remove 'keepdim' and, if the original shape must be retained, manually broadcast the reduced tensor (e.g., val = val[:, None] for axis=1)."
            ))
        
        return issues
    
    def _validate_load_patterns(self, code: str) -> List[ValidationIssue]:
        """Validate tl.load usage patterns."""
        issues = []
        
        # Pattern: tl.load(ptr, size) - this creates a scalar, not a tensor
        scalar_load_pattern = r'tl\.load\(\s*([^,]+)\s*,\s*(\w+)\s*\)'
        matches = re.finditer(scalar_load_pattern, code)
        
        for match in matches:
            ptr_expr = match.group(1).strip()
            size_var = match.group(2).strip()
            
            # Check if this looks like it should be a tensor load
            if not ptr_expr.endswith('_ptr') and 'offset' not in ptr_expr.lower():
                continue
                
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Potential scalar load pattern: tl.load({ptr_expr}, {size_var}). This creates a scalar, not a tensor.",
                suggestion=f"For tensor operations, use: tl.load({ptr_expr} + offsets, mask=mask) where offsets = tl.arange(0, BLOCK_SIZE)"
            ))
        
        # Pattern: Missing mask in tl.load
        load_without_mask = r'tl\.load\([^)]+\+[^)]*offsets[^)]*\)(?![^)]*mask)'
        matches = re.finditer(load_without_mask, code)
        
        for match in matches:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="tl.load with offsets should include mask parameter for boundary checking",
                suggestion="Add mask=mask parameter: tl.load(ptr + offsets, mask=mask)"
            ))
        
        return issues
    
    def _validate_launcher_patterns(self, code: str) -> List[ValidationIssue]:
        """Validate launcher function patterns."""
        issues = []
        
        # Pattern: Using undefined 'meta' parameter
        meta_usage = r'meta\[[\'"]\w+[\'"]\]'
        if re.search(meta_usage, code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Usage of undefined 'meta' parameter in grid lambda function",
                suggestion="Use direct values: grid = (triton.cdiv(M, BLOCK_SIZE),) instead of lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)"
            ))
        
        # Pattern: Double grid assignment
        double_grid_pattern = r'(\w+)\s*=\s*\w+\[grid\]\s*\n.*\1\[grid\]'
        if re.search(double_grid_pattern, code, re.MULTILINE):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Double grid assignment pattern detected",
                suggestion="Use either kernel[grid](...) or assign once: kernel_with_grid = kernel[grid]; kernel_with_grid(...)"
            ))
        
        return issues
    
    def _validate_deprecated_apis(self, code: str) -> List[ValidationIssue]:
        """Validate against deprecated or non-existent APIs."""
        issues = []
        
        # Pattern: tl.shared_tensor (doesn't exist in Triton 3.3.1)
        if 'tl.shared_tensor' in code:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="tl.shared_tensor() does not exist in Triton 3.3.1",
                suggestion="Use tl.zeros([shape], dtype=tl.float32) for shared memory allocation"
            ))
        
        # Pattern: Incorrect tensor slicing
        tensor_slice_pattern = r'\w+\[[^\]]*:\s*[^\]]*\]'
        if re.search(tensor_slice_pattern, code):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Complex tensor slicing may not be supported in Triton",
                suggestion="Use tl.load() and tl.store() with proper pointer arithmetic instead"
            ))
        
        return issues


class TritonAPIValidator:
    """Validates Triton API usage."""
    
    # Known Triton language functions
    VALID_TL_FUNCTIONS = {
        'program_id', 'num_programs', 'arange', 'range', 'load', 'store', 'zeros', 'ones', 'full',
        'sum', 'max', 'min', 'mean', 'dot', 'trans', 'view', 'reshape',
        'exp', 'log', 'sqrt', 'sin', 'cos', 'sigmoid', 'tanh', 'relu',
        'maximum', 'minimum', 'abs', 'floor', 'ceil', 'round',
        'atomic_add', 'atomic_max', 'atomic_min', 'atomic_and', 'atomic_or', 'atomic_xor',
        'multiple_of', 'max_contiguous', 'constexpr', 'broadcast', 'broadcast_to',
        'expand_dims', 'interleave', 'join', 'permute', 'ravel', 'split',
        'flip', 'where', 'swizzle2d', 'cdiv', 'clamp', 'div_rn', 'erf', 'exp2',
        'fdiv', 'fma', 'log2', 'rsqrt', 'sqrt_rn', 'umulhi', 'argmax', 'argmin',
        'reduce', 'xor_sum', 'associative_scan', 'cumprod', 'cumsum', 'histogram',
        'sort', 'gather', 'atomic_cas', 'atomic_xchg', 'static_range', 'softmax'
    }
    
    # Functions that don't exist in Triton but are commonly mistaken
    INVALID_TL_FUNCTIONS = {
        'thread_id': 'Use tl.program_id(axis) instead',
        'threadIdx': 'Use tl.program_id(axis) instead',
        'blockIdx': 'Use tl.program_id(axis) instead',
        'blockDim': 'Not available in Triton, use BLOCK_SIZE parameters',
        'gridDim': 'Not available in Triton, use grid configuration',
        'shared_tensor': 'Use tl.zeros([shape], dtype) instead'
    }
    
    def validate_api_usage(self, code: str) -> List[ValidationIssue]:
        """Validate Triton API usage in the code."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    issues.extend(self._validate_function_call(node))
                elif isinstance(node, ast.Attribute):
                    issues.extend(self._validate_attribute_access(node))
        
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset
            ))
        
        return issues
    
    def _validate_function_call(self, node: ast.Call) -> List[ValidationIssue]:
        """Validate a function call node."""
        issues = []
        
        if isinstance(node.func, ast.Attribute):
            # Check for tl.function_name calls
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'tl':
                func_name = node.func.attr
                
                if func_name in self.INVALID_TL_FUNCTIONS:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid Triton function: tl.{func_name}",
                        line_number=getattr(node, 'lineno', None),
                        suggestion=self.INVALID_TL_FUNCTIONS[func_name]
                    ))
                elif func_name not in self.VALID_TL_FUNCTIONS:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Unknown Triton function: tl.{func_name}",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Verify this function exists in your Triton version"
                    ))
        
        return issues
    
    def _validate_attribute_access(self, node: ast.Attribute) -> List[ValidationIssue]:
        """Validate attribute access patterns."""
        issues = []
        
        # Check for common CUDA patterns that don't work in Triton
        if isinstance(node.value, ast.Name):
            if node.value.id in ['threadIdx', 'blockIdx', 'blockDim', 'gridDim']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"CUDA built-in '{node.value.id}' not available in Triton",
                    line_number=getattr(node, 'lineno', None),
                    suggestion="Use tl.program_id(axis) for thread/block indexing"
                ))
        
        return issues


class KernelSignatureValidator:
    """Validates kernel function signatures."""
    
    def validate_signature(self, code: str, input_specs: List[TensorSpec]) -> List[ValidationIssue]:
        """Validate the kernel function signature."""
        issues = []
        
        try:
            tree = ast.parse(code)
            kernel_functions = self._find_kernel_functions(tree)
            
            if not kernel_functions:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="No @triton.jit decorated function found",
                    suggestion="Add @triton.jit decorator to your kernel function"
                ))
                return issues
            
            for func_node in kernel_functions:
                issues.extend(self._validate_function_signature(func_node, input_specs))
        
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Syntax error in signature validation: {e.msg}",
                line_number=e.lineno
            ))
        
        return issues
    
    def _find_kernel_functions(self, tree: ast.AST) -> List[ast.FunctionDef]:
        """Find functions decorated with @triton.jit."""
        kernel_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for @triton.jit decorator
                for decorator in node.decorator_list:
                    if self._is_triton_jit_decorator(decorator):
                        kernel_functions.append(node)
                        break
        
        return kernel_functions
    
    def _is_triton_jit_decorator(self, decorator: ast.expr) -> bool:
        """Check if a decorator is @triton.jit."""
        if isinstance(decorator, ast.Attribute):
            return (isinstance(decorator.value, ast.Name) and 
                   decorator.value.id == 'triton' and 
                   decorator.attr == 'jit')
        elif isinstance(decorator, ast.Name):
            return decorator.id == 'jit'  # Assuming 'from triton import jit'
        return False
    
    def _validate_function_signature(self, func_node: ast.FunctionDef, 
                                   input_specs: List[TensorSpec]) -> List[ValidationIssue]:
        """Validate a specific function signature."""
        issues = []
        
        args = func_node.args.args
        if not args:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Kernel function has no parameters",
                line_number=func_node.lineno,
                suggestion="Add at least output_ptr parameter"
            ))
            return issues
        
        # Check parameter naming conventions
        param_names = [arg.arg for arg in args]
        
        # First parameter should be output pointer
        if not param_names[0].endswith('_ptr'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"First parameter '{param_names[0]}' should be output pointer (end with '_ptr')",
                line_number=func_node.lineno,
                suggestion="Rename to 'output_ptr' or similar"
            ))
        
        # Check for constexpr parameters
        constexpr_params = []
        for arg in args:
            if arg.annotation:
                if (isinstance(arg.annotation, ast.Attribute) and
                    isinstance(arg.annotation.value, ast.Name) and
                    arg.annotation.value.id == 'tl' and
                    arg.annotation.attr == 'constexpr'):
                    constexpr_params.append(arg.arg)
        
        if not constexpr_params:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No tl.constexpr parameters found",
                line_number=func_node.lineno,
                suggestion="Add BLOCK_SIZE: tl.constexpr parameter for block size"
            ))
        
        # Validate parameter count vs input specs
        expected_min_params = 1 + len(input_specs)  # output + inputs
        if len(param_names) < expected_min_params:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Expected at least {expected_min_params} parameters (output + {len(input_specs)} inputs), got {len(param_names)}",
                line_number=func_node.lineno
            ))
        
        return issues


class MemoryAccessValidator:
    """Validates memory access patterns in kernels."""
    
    def validate_memory_access(self, code: str) -> List[ValidationIssue]:
        """Validate memory access patterns."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check for proper masking in load/store operations
            issues.extend(self._validate_load_store_operations(tree))
            
            # Check for boundary checking
            issues.extend(self._validate_boundary_checks(tree))
            
        except SyntaxError:
            # Syntax errors will be caught by other validators
            pass
        
        return issues
    
    def _validate_load_store_operations(self, tree: ast.AST) -> List[ValidationIssue]:
        """Validate tl.load and tl.store operations."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'tl'):
                    
                    if node.func.attr == 'load':
                        issues.extend(self._validate_load_call(node))
                    elif node.func.attr == 'store':
                        issues.extend(self._validate_store_call(node))
        
        return issues
    
    def _validate_load_call(self, node: ast.Call) -> List[ValidationIssue]:
        """Validate a tl.load call."""
        issues = []
        
        # Check if mask parameter is used
        has_mask = False
        for keyword in node.keywords:
            if keyword.arg == 'mask':
                has_mask = True
                break
        
        if not has_mask:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="tl.load call without mask parameter",
                line_number=getattr(node, 'lineno', None),
                suggestion="Add mask=mask parameter for boundary checking"
            ))
        
        return issues
    
    def _validate_store_call(self, node: ast.Call) -> List[ValidationIssue]:
        """Validate a tl.store call."""
        issues = []
        
        # Check if mask parameter is used
        has_mask = False
        for keyword in node.keywords:
            if keyword.arg == 'mask':
                has_mask = True
                break
        
        if not has_mask:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="tl.store call without mask parameter",
                line_number=getattr(node, 'lineno', None),
                suggestion="Add mask=mask parameter for boundary checking"
            ))
        
        return issues
    
    def _validate_boundary_checks(self, tree: ast.AST) -> List[ValidationIssue]:
        """Validate that proper boundary checks are implemented."""
        issues = []
        
        # Look for mask variable creation
        has_mask_creation = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'mask':
                        has_mask_creation = True
                        break
        
        if not has_mask_creation:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No boundary check mask found",
                suggestion="Create mask with: mask = offsets < n_elements"
            ))
        
        return issues


class TritonErrorFixer:
    """Automatically fixes common Triton compilation errors."""
    
    def fix_common_errors(self, code: str) -> tuple[str, List[str]]:
        """
        Automatically fix common Triton errors.
        Returns (fixed_code, list_of_fixes_applied).
        """
        fixes_applied = []
        fixed_code = code
        
        # Fix 1: Convert scalar tl.load to tensor tl.load when followed by axis operations
        fixed_code, scalar_fixes = self._fix_scalar_load_with_axis(fixed_code)
        fixes_applied.extend(scalar_fixes)
        
        # Fix 2: Fix launcher function meta parameter issues
        fixed_code, launcher_fixes = self._fix_launcher_meta_issues(fixed_code)
        fixes_applied.extend(launcher_fixes)
        
        # Fix 3: Fix deprecated API usage
        fixed_code, api_fixes = self._fix_deprecated_apis(fixed_code)
        fixes_applied.extend(api_fixes)
        
        # Fix 4: Add missing boundary checks
        fixed_code, boundary_fixes = self._fix_missing_boundary_checks(fixed_code)
        fixes_applied.extend(boundary_fixes)
        
        # NEW Fix 5: Remove unsupported 'keepdim' argument from reductions
        fixed_code, keepdim_fixes = self._fix_keepdim_keyword(fixed_code)
        fixes_applied.extend(keepdim_fixes)
        
        return fixed_code, fixes_applied
    
    def _fix_scalar_load_with_axis(self, code: str) -> tuple[str, List[str]]:
        """Fix the specific scalar load + axis=0 error pattern."""
        fixes = []
        
        # Pattern: var = tl.load(ptr, size) followed by tl.max(var, axis=0)
        # This is the exact error from the logs
        pattern = r'(\w+)\s*=\s*tl\.load\(([^,]+),\s*(\w+)\s*\)\s*\n.*?tl\.(max|min|sum|mean)\(\s*\1\s*,\s*axis\s*=\s*0\s*\)'
        
        def replace_scalar_load(match):
            var_name = match.group(1)
            ptr_expr = match.group(2).strip()
            size_var = match.group(3).strip()
            reduction_func = match.group(4)
            
            # Generate the fixed version
            fixed_pattern = f"""# Fixed: Convert scalar load to tensor load for axis operations
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < {size_var}
    {var_name} = tl.load({ptr_expr} + offsets, mask=mask, other=0.0)
    # Now can use axis=0 on tensor
    {var_name}_result = tl.{reduction_func}({var_name}, axis=0)"""
            
            fixes.append(f"Fixed scalar load pattern for variable '{var_name}' - converted to tensor load with proper offsets")
            return fixed_pattern
        
        fixed_code = re.sub(pattern, replace_scalar_load, code, flags=re.MULTILINE | re.DOTALL)
        
        # Also fix the specific softmax pattern from the error logs
        softmax_pattern = r'row_data\s*=\s*tl\.load\(row_start_ptr,\s*N\s*\)\s*\n\s*row_max_val\s*=\s*tl\.max\(row_data,\s*axis\s*=\s*0\s*\)'
        
        def fix_softmax_pattern(match):
            fixes.append("Fixed softmax row_data scalar load pattern - converted to proper tensor operations")
            return """# Fixed: Proper tensor loading for softmax
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_data = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    row_max_val = tl.max(row_data, axis=0)"""
        
        fixed_code = re.sub(softmax_pattern, fix_softmax_pattern, fixed_code, flags=re.MULTILINE)
        
        return fixed_code, fixes
    
    def _fix_launcher_meta_issues(self, code: str) -> tuple[str, List[str]]:
        """Fix launcher function meta parameter issues."""
        fixes = []
        
        # Fix: grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
        meta_pattern = r'grid\s*=\s*lambda\s+meta:\s*\([^)]*meta\[[\'"](\w+)[\'"]\][^)]*\)'
        
        def fix_meta_usage(match):
            param_name = match.group(1)
            fixes.append(f"Fixed undefined meta parameter usage - replaced with direct {param_name} value")
            return f"grid = (triton.cdiv(M, {param_name}),)"
        
        fixed_code = re.sub(meta_pattern, fix_meta_usage, code)
        
        # Fix double grid assignment
        double_grid_pattern = r'(\w+)\s*=\s*(\w+)\[grid\]\s*\n\s*\1\[grid\]'
        
        def fix_double_grid(match):
            kernel_var = match.group(1)
            kernel_name = match.group(2)
            fixes.append("Fixed double grid assignment pattern")
            return f"{kernel_name}[grid]"
        
        fixed_code = re.sub(double_grid_pattern, fix_double_grid, fixed_code, flags=re.MULTILINE)
        
        return fixed_code, fixes
    
    def _fix_deprecated_apis(self, code: str) -> tuple[str, List[str]]:
        """Fix deprecated API usage."""
        fixes = []
        fixed_code = code
        
        # Fix tl.shared_tensor usage - more comprehensive pattern
        if 'tl.shared_tensor' in code:
            # Pattern 1: tl.shared_tensor([shape], dtype=type)
            pattern1 = r'tl\.shared_tensor\(\[([^\]]+)\],\s*dtype\s*=\s*([^)]+)\)'
            if re.search(pattern1, fixed_code):
                fixed_code = re.sub(pattern1, r'tl.zeros([\1], dtype=\2)', fixed_code)
                fixes.append("Fixed tl.shared_tensor() usage - replaced with tl.zeros()")
            
            # Pattern 2: tl.shared_tensor([shape], type) - without dtype keyword
            pattern2 = r'tl\.shared_tensor\(\[([^\]]+)\],\s*([^)]+)\)'
            if re.search(pattern2, fixed_code):
                fixed_code = re.sub(pattern2, r'tl.zeros([\1], dtype=\2)', fixed_code)
                fixes.append("Fixed tl.shared_tensor() usage - replaced with tl.zeros()")
            
            # Pattern 3: Any remaining tl.shared_tensor calls
            if 'tl.shared_tensor' in fixed_code:
                fixed_code = fixed_code.replace('tl.shared_tensor', 'tl.zeros')
                fixes.append("Fixed remaining tl.shared_tensor() usage - replaced with tl.zeros()")
        
        return fixed_code, fixes
    
    def _fix_missing_boundary_checks(self, code: str) -> tuple[str, List[str]]:
        """Add missing boundary checks to tl.load/store operations."""
        fixes = []
        
        # Find tl.load calls without mask
        load_pattern = r'tl\.load\(([^)]+\+[^)]*offsets[^)]*)\)(?![^)]*mask)'
        
        def add_mask_to_load(match):
            load_args = match.group(1)
            fixes.append("Added missing mask parameter to tl.load()")
            return f"tl.load({load_args}, mask=mask, other=0.0)"
        
        fixed_code = re.sub(load_pattern, add_mask_to_load, code)
        
        # Find tl.store calls without mask
        store_pattern = r'tl\.store\(([^)]+\+[^)]*offsets[^)]*),\s*([^)]+)\)(?![^)]*mask)'
        
        def add_mask_to_store(match):
            ptr_args = match.group(1)
            value_arg = match.group(2)
            fixes.append("Added missing mask parameter to tl.store()")
            return f"tl.store({ptr_args}, {value_arg}, mask=mask)"
        
        fixed_code = re.sub(store_pattern, add_mask_to_store, fixed_code)
        
        return fixed_code, fixes

    # NEW helper -------------------------------------------------------------
    def _fix_keepdim_keyword(self, code: str) -> tuple[str, List[str]]:
        """Remove unsupported keepdim keyword from Triton reduction ops and optionally broadcast back."""
        fixes = []
        # Regex to capture reduction call with keepdim, preserving before/after parts of the call
        pattern = r"(tl\.(?:max|min|sum|mean|argmax|argmin|logsumexp)\([^)]*?)(,\s*axis\s*=\s*[^,)]*)?\s*,\s*keepdim\s*=\s*(?:True|False|1|0)([^)]*\))"
        
        def repl(match):
            prefix = match.group(1)
            axis_part = match.group(2) or ""
            suffix = match.group(3)
            fixes.append("Removed unsupported 'keepdim' argument from a reduction op and left note to broadcast manually if required")
            return f"{prefix}{axis_part}{suffix}  # NOTE: broadcast manually if original dims needed"
        
        fixed_code = re.sub(pattern, repl, code)
        return fixed_code, fixes


class KernelValidator:
    """Main kernel validator that orchestrates all validation checks."""
    
    def __init__(self):
        self.api_validator = TritonAPIValidator()
        self.signature_validator = KernelSignatureValidator()
        self.memory_validator = MemoryAccessValidator()
        self.triton_error_pattern_validator = TritonErrorPatternValidator()
        self.triton_error_fixer = TritonErrorFixer()
    
    def validate(self, kernel_code: str, input_specs: List[TensorSpec]) -> ValidationResult:
        """Perform comprehensive validation of kernel code."""
        logger.info("Starting kernel validation")
        
        all_issues = []
        
        # Basic syntax check
        try:
            ast.parse(kernel_code)
        except SyntaxError as e:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset
            ))
            # If there's a syntax error, skip other validations
            return ValidationResult(is_valid=False, issues=all_issues)
        
        # API usage validation
        all_issues.extend(self.api_validator.validate_api_usage(kernel_code))
        
        # Signature validation
        all_issues.extend(self.signature_validator.validate_signature(kernel_code, input_specs))
        
        # Memory access validation
        all_issues.extend(self.memory_validator.validate_memory_access(kernel_code))
        
        # Additional semantic checks
        all_issues.extend(self._validate_imports(kernel_code))
        all_issues.extend(self._validate_launcher_function(kernel_code))
        
        # Critical error pattern detection
        all_issues.extend(self.triton_error_pattern_validator.validate_error_patterns(kernel_code))
        
        # Attempt to fix common errors if there are critical issues
        has_critical_errors = any(issue.severity == ValidationSeverity.ERROR for issue in all_issues)
        if has_critical_errors:
            fixed_code, fixes_applied = self.triton_error_fixer.fix_common_errors(kernel_code)
            if fixes_applied:
                # Add info about fixes applied
                for fix_description in fixes_applied:
                    all_issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Auto-fix applied: {fix_description}"
                    ))
                
                # Re-validate the fixed code (avoid infinite recursion by not auto-fixing again)
                logger.info("Re-validating auto-fixed code")
                fixed_issues = []
                
                # Re-run all validations on fixed code
                fixed_issues.extend(self.api_validator.validate_api_usage(fixed_code))
                fixed_issues.extend(self.signature_validator.validate_signature(fixed_code, input_specs))
                fixed_issues.extend(self.memory_validator.validate_memory_access(fixed_code))
                fixed_issues.extend(self._validate_imports(fixed_code))
                fixed_issues.extend(self._validate_launcher_function(fixed_code))
                fixed_issues.extend(self.triton_error_pattern_validator.validate_error_patterns(fixed_code))
                
                fixed_has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in fixed_issues)
                if not fixed_has_errors:
                    logger.info("Auto-fix successful - kernel is now valid")
                    # Return successful validation with fixed code
                    return ValidationResult(
                        is_valid=True,
                        issues=all_issues + fixed_issues,  # Include original issues + fix info
                        fixed_code=fixed_code
                    )
                else:
                    logger.info("Auto-fix applied but some errors remain")
                    # Still return the fixed code even if some errors remain
                    # This allows the system to use the partially fixed code
                    return ValidationResult(
                        is_valid=False,
                        issues=all_issues + fixed_issues,
                        fixed_code=fixed_code
                    )
        
        # Determine if kernel is valid (no errors)
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in all_issues)
        is_valid = not has_errors
        
        logger.info("Kernel validation complete: %s (%d issues)", 
                   "VALID" if is_valid else "INVALID", len(all_issues))
        
        return ValidationResult(is_valid=is_valid, issues=all_issues)
    
    def _validate_imports(self, code: str) -> List[ValidationIssue]:
        """Validate that necessary imports are present."""
        issues = []
        
        required_imports = ['triton', 'triton.language']
        
        for required in required_imports:
            if required not in code:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing import: {required}",
                    suggestion=f"Add 'import {required}' or 'import {required} as tl'"
                ))
        
        return issues
    
    def _validate_launcher_function(self, code: str) -> List[ValidationIssue]:
        """Validate that a launcher function is present."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Look for launcher function (function that calls the kernel)
            has_launcher = False
            kernel_calls = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    # Look for kernel[grid](...) pattern
                    if isinstance(node.value, ast.Name):
                        kernel_calls.append(node.value.id)
                        has_launcher = True
            
            if not has_launcher:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="No launcher function found",
                    suggestion="Add a launcher function that calls the kernel with appropriate grid"
                ))
        
        except SyntaxError:
            # Syntax errors handled elsewhere
            pass
        
        return issues
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a human-readable summary of validation results."""
        if result.is_valid:
            summary = "✓ Kernel validation PASSED"
        else:
            summary = "✗ Kernel validation FAILED"
        
        if result.issues:
            summary += f" ({len(result.issues)} issues found)"
            
            error_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.WARNING)
            
            if error_count > 0:
                summary += f"\n  Errors: {error_count}"
            if warning_count > 0:
                summary += f"\n  Warnings: {warning_count}"
        
        return summary 