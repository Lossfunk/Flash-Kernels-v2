from __future__ import annotations

from pathlib import Path
from google.adk.tools.function_tool import FunctionTool
from utils.genai_client import chat

from agents.base import BaseAgent
from agents.contracts import CorrectnessReasonerIn, CorrectnessReasonerOut
from agents.correctness_reasoner.prompts import SYSTEM_PROMPT

from utils.logging_utils import get_logger

logger = get_logger("CorrectnessReasonerAgent")


def _analyze_kernel_calling_pattern(payload: CorrectnessReasonerIn) -> CorrectnessReasonerOut:
    logger.info("Analyzing kernel source to determine correct calling pattern")
    
    # Read the kernel source
    kernel_source = ""
    try:
        # Check if it looks like a file path (starts with / or contains .py/.ptx and is reasonably short)
        if (payload.kernel_source_path.startswith('/') or 
            payload.kernel_source_path.endswith(('.py', '.ptx')) or
            'tmp' in payload.kernel_source_path) and len(payload.kernel_source_path) < 500:
            # Likely a file path, try to read from file
            try:
                kernel_source_path = Path(payload.kernel_source_path)
                if kernel_source_path.exists() and kernel_source_path.is_file():
                    kernel_source = kernel_source_path.read_text()
                    logger.debug("Read kernel source from file %s (%d chars)", payload.kernel_source_path, len(kernel_source))
                else:
                    # File doesn't exist, treat as source code
                    kernel_source = payload.kernel_source_path
                    logger.debug("File doesn't exist, treating as kernel source code directly (%d chars)", len(kernel_source))
            except (OSError, ValueError) as path_error:
                # Failed to read file, treat as source code
                kernel_source = payload.kernel_source_path
                logger.debug("File read failed (%s), treating as kernel source code directly (%d chars)", 
                            str(path_error), len(kernel_source))
        else:
            # Looks like source code (too long or doesn't look like a path), use directly
            kernel_source = payload.kernel_source_path
            logger.debug("Treating as kernel source code directly (%d chars)", len(kernel_source))
        
        if not kernel_source.strip():
            raise ValueError("Kernel source is empty.")

    except Exception as e:
        logger.error("Failed to read kernel source: %s", e)
        return CorrectnessReasonerOut(
            calling_pattern="# Error: Could not read kernel source",
            grid_config="# Error",
            kernel_args=["# Error reading source"],
            explanation=f"Failed to read kernel source: {e}"
        )
    
    # Parse kernel signature deterministically
    try:
        import re
        import ast
        
        # Extract the kernel function definition
        kernel_match = re.search(r'@triton\.jit\s*\ndef\s+(\w+)\s*\((.*?)\):', kernel_source, re.DOTALL)
        if not kernel_match:
            raise ValueError("Could not find @triton.jit kernel function")
        
        kernel_name = kernel_match.group(1)
        params_str = kernel_match.group(2)
        
        # Parse parameters
        param_names = []
        constexpr_params = []
        
        # Simple parameter parsing (handles most common cases)
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            
            # Handle type annotations and defaults
            param_name = param.split(':')[0].strip()
            if '=' in param_name:
                param_name = param_name.split('=')[0].strip()
            
            param_names.append(param_name)
            
            # Check if it's a constexpr parameter
            if 'tl.constexpr' in param or 'constexpr' in param:
                constexpr_params.append(param_name)
        
        logger.info("Parsed kernel '%s' with parameters: %s", kernel_name, param_names)
        logger.info("Constexpr parameters: %s", constexpr_params)
        
        # Determine input/output tensor mapping
        input_specs = payload.input_specs
        expected_shape = payload.expected_output_shape
        
        # Build argument list intelligently
        args = []
        
        # Map tensor arguments (usually first few parameters)
        tensor_params = [p for p in param_names if p.endswith('_ptr') or 'ptr' in p.lower()]
        
        # Typically: output_ptr, input1_ptr, input2_ptr, ...
        if len(tensor_params) >= 1:
            args.append("out")  # Output tensor
        if len(tensor_params) >= 2:
            args.extend([f"input{i}" for i in range(len(input_specs))])
        
        # Derive M, N, K from input_specs primarily
        derived_M, derived_N, derived_K = None, None, None
        if input_specs and len(input_specs) > 0:
            if len(input_specs[0].shape) >= 1: # M is from input0.shape[0]
                derived_M = input_specs[0].shape[0]
            if len(input_specs[0].shape) >= 2: # K is from input0.shape[1]
                derived_K = input_specs[0].shape[1]
            
            if len(input_specs) > 1 and len(input_specs[1].shape) >= 2: # N is from input1.shape[1]
                # K can also be input_specs[1].shape[0], ensure consistency if both available
                if derived_K is not None and derived_K != input_specs[1].shape[0]:
                    logger.warning(f"K dimension mismatch: input0.shape[1] ({derived_K}) vs input1.shape[0] ({input_specs[1].shape[0]})")
                    # Potentially prioritize K from input0 or raise error, for now, log and proceed with input0's K
                elif derived_K is None:
                    derived_K = input_specs[1].shape[0]
                derived_N = input_specs[1].shape[1]
        
        # Add dimension parameters
        # Use derived M, N, K if expected_shape didn't provide them or is absent
        m_to_add, n_to_add, k_to_add = None, None, None

        if expected_shape and len(expected_shape) == 2:
            m_to_add, n_to_add = expected_shape[0], expected_shape[1]
            if input_specs and len(input_specs) >= 1 and len(input_specs[0].shape) >= 2:
                k_to_add = input_specs[0].shape[1] # K from input0.shape[1]
        
        # Override with derived values if any were None from expected_shape logic
        if m_to_add is None and derived_M is not None: m_to_add = derived_M
        if n_to_add is None and derived_N is not None: n_to_add = derived_N
        if k_to_add is None and derived_K is not None: k_to_add = derived_K

        param_check_M = 'M' in param_names or any('m' == p.lower() or '_m' in p.lower() for p in param_names)
        param_check_N = 'N' in param_names or any('n' == p.lower() or '_n' in p.lower() for p in param_names)
        param_check_K = 'K' in param_names or any('k' == p.lower() or '_k' in p.lower() for p in param_names)

        if param_check_M and m_to_add is not None:
            args.append(str(m_to_add))
        if param_check_N and n_to_add is not None:
            args.append(str(n_to_add))
        if param_check_K and k_to_add is not None:
            args.append(str(k_to_add))
        
        # Fallback for single dimension N if M,K are not found/applicable (e.g. vector op)
        if not (param_check_M and param_check_K) and param_check_N and n_to_add is not None and str(n_to_add) not in args:
            if expected_shape and len(expected_shape) == 1: # Vector operation
                 args.append(str(expected_shape[0]))
            elif derived_N is not None: # If N was derived and not added
                 args.append(str(derived_N))

        # Add stride parameters
        stride_params = [p for p in param_names if 'stride' in p.lower()]
        for stride_param in stride_params:
            # Map stride parameters to tensor strides
            if 'out' in stride_param.lower() or 'c' in stride_param.lower(): # Output tensor C
                # stride_cm, stride_cn
                if '0' in stride_param or 'm' in stride_param.lower() or stride_param.endswith("cm") or stride_param.endswith("mc"): # M dimension stride for C
                    args.append("out.stride(0)")
                elif '1' in stride_param or 'n' in stride_param.lower() or stride_param.endswith("cn") or stride_param.endswith("nc"): # N dimension stride for C
                    args.append("out.stride(1)")
                else: # Fallback for output strides if specific dim not identified
                    args.append("out.stride(0)") # Default to first stride
            elif 'a' in stride_param.lower() or ('input' in stride_param.lower() and '0' in stride_param): # Input tensor A
                # stride_am, stride_ak
                if '0' in stride_param or 'm' in stride_param.lower() or stride_param.endswith("am") or stride_param.endswith("ma"): # M dimension stride for A
                    args.append("input0.stride(0)")
                elif '1' in stride_param or 'k' in stride_param.lower() or stride_param.endswith("ak") or stride_param.endswith("ka"): # K dimension stride for A
                    args.append("input0.stride(1)")
                else: # Fallback for input0 strides
                    args.append("input0.stride(0)")
            elif 'b' in stride_param.lower() or ('input' in stride_param.lower() and '1' in stride_param): # Input tensor B
                # stride_bk, stride_bn
                # Correct stride_bk and stride_bn
                if '0' in stride_param or 'k' in stride_param.lower() or stride_param.endswith("bk") or stride_param.endswith("kb"): # K dimension stride for B
                    args.append("input1.stride(0)")
                elif '1' in stride_param or 'n' in stride_param.lower() or stride_param.endswith("bn") or stride_param.endswith("nb"): # N dimension stride for B
                    args.append("input1.stride(1)")
                else: # Fallback for input1 strides
                     args.append("input1.stride(0)") # Default to first stride for B, e.g. K
            else:
                # Fallback for unknown stride params, try to guess based on order or default
                # This part might need more sophisticated logic if generic strides appear often
                if any(x in stride_param for x in ["out", "c_"]): args.append("out.stride(0)")
                elif any(x in stride_param for x in ["in0", "a_"]): args.append("input0.stride(0)")
                elif any(x in stride_param for x in ["in1", "b_"]): args.append("input1.stride(0)")
                else: args.append("1")  # Default stride if unmapped

        # Add BLOCK_SIZE_* parameters
        # These are typically constexpr in Triton but must be passed if in the kernel signature.
        # The values (e.g., 32) are often defaults used during compilation.
        # For robustness, these values could be part of `payload` or a config.
        # Here, we use a common default if the parameter name matches.
        block_size_default = "32" # Default value from logs
        for p_name in param_names:
            if p_name.startswith("BLOCK_SIZE_") and p_name not in args:
                # Check if this param was already added by name (e.g. if M, N, K were block sizes)
                # This is a simple check; more robust would be to track added args by their meaning.
                already_added = False
                for arg_val_str in args:
                    if arg_val_str == p_name: # If the string itself was added (unlikely for block sizes)
                        already_added = True
                        break
                if not already_added:
                     # Only add if not already present by some other logic (e.g. if M/N/K are sizes)
                     # and if it's a known BLOCK_SIZE pattern.
                    if p_name in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "BLOCK_SIZE"]:
                         args.append(block_size_default)
                         logger.info(f"Added {p_name}={block_size_default} to kernel_args")
        
        # Generate grid configuration
        grid_M_val, grid_N_val = m_to_add, n_to_add # Use the determined M, N for grid

        if grid_M_val is not None and grid_N_val is not None: # Matrix operation
            grid_config = f"(triton.cdiv({grid_M_val}, BLOCK_SIZE_M), triton.cdiv({grid_N_val}, BLOCK_SIZE_N))"
        elif grid_N_val is not None: # Vector operation if only N is available (or M was not applicable)
            grid_config = f"(triton.cdiv({grid_N_val}, BLOCK_SIZE),)" # Assumes BLOCK_SIZE if M not used
        elif expected_shape and len(expected_shape) == 1: # Original fallback for 1D expected_shape
            N_vec_grid = expected_shape[0]
            grid_config = f"(triton.cdiv({N_vec_grid}, BLOCK_SIZE),)"
        else: # Fallback grid
            grid_config = "(triton.cdiv(out.numel(), 256),)"
        
        explanation = f"""
Deterministic analysis of kernel '{kernel_name}':
- Parameters: {param_names}
- Constexpr params: {constexpr_params}
- Tensor parameters: {tensor_params}
- Stride parameters: {stride_params}
- Input shapes: {[spec.shape for spec in input_specs]}
- Expected output shape: {expected_shape}

The kernel appears to be a {'matrix' if len(expected_shape) == 2 else 'vector'} operation.
Arguments mapped based on common Triton kernel patterns.
"""
        
        return CorrectnessReasonerOut(
            calling_pattern=f"{kernel_name}[{grid_config}]({', '.join(args)})",
            grid_config=grid_config,
            kernel_args=args,
            explanation=explanation.strip()
        )
        
    except Exception as e:
        logger.error("Failed to parse kernel signature: %s", e)
        # Fallback to LLM analysis if parsing fails
        return _fallback_llm_analysis(payload, kernel_source)

def _fallback_llm_analysis(payload: CorrectnessReasonerIn, kernel_source: str) -> CorrectnessReasonerOut:
    """Fallback to LLM analysis if deterministic parsing fails."""
    logger.info("Falling back to LLM analysis")
    
    # Prepare the analysis prompt
    input_specs_str = "\n".join([
        f"  Input {i}: shape={spec.shape}, dtype={spec.dtype}" 
        for i, spec in enumerate(payload.input_specs)
    ])
    
    previous_attempts_str = ""
    if payload.previous_reasoning_attempts:
        previous_attempts_str = "\n\nPrevious Reasoning Attempts and Their Outcomes:\n"
        for i, attempt in enumerate(payload.previous_reasoning_attempts):
            previous_attempts_str += (
                f"Attempt {i+1}:\n"
                f"  Suggested Grid: {attempt.suggested_grid}\n"
                f"  Suggested Args: {attempt.suggested_args}\n"
                f"  Error Received: {attempt.error_received}\n"
            )

    user_message = f"""
Kernel Source Code:
```python
{kernel_source}
```

Input Specifications:
{input_specs_str}

Expected Output Shape: {payload.expected_output_shape}

Error Message from Failed Execution:
{payload.error_message}
{previous_attempts_str}

Please analyze this kernel and provide the correct calling pattern that should work with these inputs.
Focus on the exact parameter names in the kernel signature and map them to the available tensors and values.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.debug("Sending kernel analysis request to LLM")
    try:
        response = chat(messages, temperature=0.0)
        logger.info("Received kernel calling pattern analysis")
        
        # Parse the response to extract the required fields
        lines = response.strip().split('\n')
        
        calling_pattern = ""
        grid_config = ""
        kernel_args = []
        explanation = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("calling_pattern:") or line.startswith("1. calling_pattern:"):
                current_section = "calling_pattern"
                calling_pattern = line.split(":", 1)[1].strip()
            elif line.startswith("grid_config:") or line.startswith("2. grid_config:"):
                current_section = "grid_config"
                grid_config = line.split(":", 1)[1].strip()
            elif line.startswith("kernel_args:") or line.startswith("3. kernel_args:"):
                current_section = "kernel_args"
                args_text_value = line.split(":", 1)[1].strip()
                if not args_text_value:
                    kernel_args = []
                else:
                    try:
                        import ast
                        parsed_args = ast.literal_eval(args_text_value)
                        if isinstance(parsed_args, list) and all(isinstance(item, str) for item in parsed_args):
                            kernel_args = parsed_args
                        else:
                            logger.warning(f"kernel_args was not a list of strings after parsing: {parsed_args}")
                            kernel_args = [args_text_value]
                    except (ValueError, SyntaxError, TypeError) as e_parse:
                        logger.warning(f"Failed to parse kernel_args string '{args_text_value}' as a list: {e_parse}. Using raw string as fallback.")
                        kernel_args = [args_text_value]
            elif line.startswith("explanation:") or line.startswith("4. explanation:"):
                current_section = "explanation"
                explanation = line.split(":", 1)[1].strip()
            elif current_section == "calling_pattern" and line:
                calling_pattern += "\n" + line
            elif current_section == "grid_config" and line:
                grid_config += "\n" + line.strip()
            elif current_section == "explanation" and line:
                explanation += "\n" + line
        
        # Ensure grid_config is a single condensed string
        grid_config = grid_config.replace('\n', ' ').strip()

        # Final check for kernel_args to ensure it is a list
        if not isinstance(kernel_args, list):
            logger.warning(f"kernel_args ended up not being a list ('{kernel_args}'), wrapping in a list.")
            kernel_args = [str(kernel_args)] if kernel_args else []
        
        return CorrectnessReasonerOut(
            calling_pattern=calling_pattern,
            grid_config=grid_config,
            kernel_args=kernel_args,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error("Failed to analyze kernel calling pattern: %s", e)
        return CorrectnessReasonerOut(
            calling_pattern="# Error: LLM analysis failed",
            grid_config="# Error",
            kernel_args=["# Error in analysis"],
            explanation=f"Failed to analyze kernel: {e}"
        )


correctness_reasoner_tool = FunctionTool(_analyze_kernel_calling_pattern)


class CorrectnessReasonerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="correctness_reasoner",
            description="Analyzes kernel source code to determine correct calling patterns for execution.",
            tools=[correctness_reasoner_tool]
        )

    async def analyze(self, payload: CorrectnessReasonerIn) -> CorrectnessReasonerOut:
        return _analyze_kernel_calling_pattern(payload) 