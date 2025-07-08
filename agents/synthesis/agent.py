from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool
from utils.genai_client import chat

from agents.base import BaseAgent
from agents.contracts import SynthIn, SynthOut

# Import the new structured synthesis system
from agents.synthesis.structured_generator import StructuredKernelGenerator
from agents.synthesis.kernel_validator import KernelValidator

import os
import json
import dotenv
import tempfile
import uuid
from pathlib import Path
import re
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from utils.logging_utils import get_logger

dotenv.load_dotenv()

logger = get_logger("SynthesisAgent")


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = None


def _synthesize_kernel(payload: SynthIn) -> SynthOut:
    """Generate Triton kernel source via structured generation with validation."""
    logger.info("Starting structured kernel synthesis")
    code = None

    # Try structured generation first
    try:
        generator = StructuredKernelGenerator()
        validator = KernelValidator()
        
        # Generate kernel using structured approach
        logger.info("Attempting structured kernel generation")
        result = generator.generate(payload)
        
        # Validate the generated kernel
        validation_result = validator.validate(result.kernel_src, payload.input_specs)
        
        if validation_result.is_valid:
            logger.info("Structured generation successful - kernel is valid")
            # Use fixed code if available, otherwise use original
            code = validation_result.fixed_code if validation_result.fixed_code else result.kernel_src
            if validation_result.fixed_code:
                logger.info("Using auto-fixed kernel code")
        else:
            # Log validation issues
            logger.warning("Structured generation produced kernel with issues:")
            error_messages_for_hint = ["Kernel static validation issues found:"]
            for issue in validation_result.issues:
                if issue.severity.value == "error":
                    logger.error("  ERROR: %s (Line: %s)", issue.message, issue.line_number)
                else:
                    logger.warning("  WARNING: %s (Line: %s)", issue.message, issue.line_number)
                error_messages_for_hint.append(f"- {issue.severity.value.upper()}: {issue.message} (Line: {issue.line_number})")
            
            critical_errors = [i for i in validation_result.issues if i.severity.value == "error"]
            if critical_errors:
                logger.warning("Critical errors found in structured kernel, preparing for LLM fallback with this kernel as context.")
                payload.previous_kernel_src = result.kernel_src
                
                # Combine new validation errors with any existing error_hint
                new_error_hint_str = "\n".join(error_messages_for_hint)
                if payload.error_hint:
                    payload.error_hint = f"Previous errors:\n{payload.error_hint}\n\nNew validation errors:\n{new_error_hint_str}"
                else:
                    payload.error_hint = new_error_hint_str
                
                return _fallback_llm_synthesis(payload)
            else:
                logger.info("Only warnings found in structured kernel, proceeding with it.")
                code = result.kernel_src
                
    except Exception as e:
        logger.error("Structured generation failed: %s", e, exc_info=True)
        logger.info("Falling back to LLM synthesis")
        # Add the exception to error_hint for the fallback
        if payload.error_hint:
            payload.error_hint += f"\nStructured generation failed with: {str(e)}"
        else:
            payload.error_hint = f"Structured generation failed with: {str(e)}"
        return _fallback_llm_synthesis(payload)

    if code is None: # Should not happen if logic is correct, but as a safeguard
        logger.error("Code was not generated after structured attempt and no fallback was triggered. This is a bug.")
        # Fallback as a last resort if code is still None
        return _fallback_llm_synthesis(payload)

    logger.info("Kernel synthesis complete | length=%d chars", len(code))
    synth_output = SynthOut(kernel_src=code)
    
    # Save synthesis output
    try:
        output_dir = Path("tmp_synthesis_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use uuid for a unique filename
        unique_filename = f"synthesis_output_{uuid.uuid4()}.json"
        filename = output_dir / unique_filename

        with open(filename, 'w') as f:
            f.write(synth_output.model_dump_json(indent=2))
        
        logger.info(f"Synthesis output successfully saved to {filename.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save synthesis output to JSON: {e}", exc_info=True)

    return synth_output


def _perform_direct_llm_synthesis(payload: SynthIn) -> SynthOut:
    """Performs a direct LLM synthesis call using the prompt prepared in research_context."""
    logger.info("Performing direct LLM synthesis via _perform_direct_llm_synthesis")

    # Import system prompt for LLM
    from agents.synthesis.prompts import SYSTEM_PROMPT

    if not payload.research_context:
        logger.error("Direct LLM synthesis called without research_context (prompt).")
        raise ValueError("Cannot perform direct LLM synthesis without a prompt in research_context.")

    prompt_parts = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": payload.research_context,  # research_context is the fully prepared prompt
        },
    ]

    try:
        logger.debug("Prompt sent to LLM (via direct call): %s", payload.research_context[:500] + "...")
        raw_llm_output = chat(prompt_parts)
        logger.debug(f"Raw LLM output received (len={len(raw_llm_output)}, direct call):\n{raw_llm_output[:1000]}...")

        # Extract Python code from markdown block
        match = re.search(r"```python\n(.*?)\n```", raw_llm_output, re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
            logger.info("Extracted Python code from markdown block (direct call).")
        else:
            # Try to find code even if the end backticks are missing, or if it's not in a block
            match_start_only = re.search(r"```python\n(.*?)", raw_llm_output, re.DOTALL)
            if match_start_only:
                extracted_code = match_start_only.group(1).strip()
                logger.info("Extracted Python code from partial markdown block (start ticks found, direct call).")
            else:
                extracted_code = raw_llm_output.strip()
                logger.info("No markdown block found, using stripped raw output as code (direct call).")
        
        code = extracted_code

    except Exception as e:
        logger.error("Direct LLM synthesis call failed with an unexpected error: %s", e, exc_info=True)
        raise RuntimeError(f"Direct LLM synthesis call failed: {e}") from e # Wrap other errors

    return SynthOut(kernel_src=code)


def _fallback_llm_synthesis(payload: SynthIn) -> SynthOut:
    """Fallback to LLM-based synthesis when structured generation fails."""
    logger.info("Using LLM fallback synthesis")
    
    # Import system prompt for LLM fallback
    from agents.synthesis.prompts import SYSTEM_PROMPT
    
    prompt_parts = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": _build_user_prompt(payload),
        },
    ]

    try:
        user_prompt_for_log = prompt_parts[1]["content"]
        logger.debug("Prompt sent to LLM (fallback): %s", user_prompt_for_log[:500] + "...")
        raw_llm_output = chat(prompt_parts)
        logger.debug(f"Raw LLM output received (len={len(raw_llm_output)}, fallback):\n{raw_llm_output[:1000]}...")

        # Extract Python code from markdown block
        match = re.search(r"```python\n(.*?)\n```", raw_llm_output, re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
            logger.info("Extracted Python code from markdown block (fallback).")
        else:
            match_start_only = re.search(r"```python\n(.*?)", raw_llm_output, re.DOTALL)
            if match_start_only:
                extracted_code = match_start_only.group(1).strip()
                logger.info("Extracted Python code from partial markdown block (start ticks found, fallback).")
            else:
                extracted_code = raw_llm_output.strip()
                logger.info("No markdown block found, using stripped raw output as code (fallback).")
        
        code = extracted_code

    except Exception as e:
        logger.error("LLM synthesis failed with an unexpected error: %s", e, exc_info=True)
        raise RuntimeError(f"LLM synthesis failed: {e}") from e

    return SynthOut(kernel_src=code)


def _build_user_prompt(payload: SynthIn) -> str:
    """Build user prompt for LLM fallback."""
    specs_json = json.dumps([s.model_dump() for s in payload.input_specs], indent=2)
    
    # Initialize sections
    compiler_hint_section = ""
    correctness_feedback_section = ""
    critical_error_intro = ""
    task_focus = ""

    # --- Process Correctness Hint ---
    if payload.correctness_hint:
        if isinstance(payload.correctness_hint, dict):
            # Handle dictionary format (legacy)
            error_type = payload.correctness_hint.get("error_type")
            error_message = payload.correctness_hint.get("error_message", "N/A")

            if error_type == "kernelbench_runtime_error":
                specific_runtime_error = payload.correctness_hint.get("specific_runtime_error", "N/A")
                critical_error_intro = (
                    "CRITICAL ISSUE: The kernel COMPILED successfully, but FAILED during KernelBench validation "
                    f"due to a specific RUNTIME ERROR: {specific_runtime_error}\\n"
                    "This runtime error is the PRIMARY problem to solve. It likely originates from the launcher function "
                    "or how the kernel interfaces with the testing environment (e.g., an incorrect method name or class structure expected by KernelBench)."
                )
                correctness_feedback_section = (
                    f"\\n\\n--- KernelBench Runtime Error Details ---\\n"
                    f"- Error Type: {error_type}\\n"
                    f"- Message: {error_message}\\n"
                    f"- Specific Runtime Error: {specific_runtime_error}\\n"
                    f"You MUST address this runtime error. Numerical stability details below might be secondary or misleading until this is fixed."
                )
                # Add other relevant fields if they exist and are not "N/A"
                for key, value in payload.correctness_hint.items():
                    if key not in ["error_type", "error_message", "specific_runtime_error", "validation_method"] and value and "N/A" not in str(value):
                        correctness_feedback_section += f"- {key.replace('_', ' ').title()}: {value}\\n"
                task_focus = "DEBUG and FIX the provided Triton kernel, focusing primarily on resolving the specified KERNELBENCH RUNTIME ERROR in the launcher or its integration. Ensure the overall kernel logic remains correct."

            elif error_type == "numerical_stability_error":
                critical_error_intro = (
                    "IMPORTANT: The kernel COMPILED successfully but FAILED numerical correctness checks."
                )
                correctness_feedback_section = "\\n\\n--- Numerical Stability Feedback ---\\n"
                correctness_feedback_section += f"- Issue Type: {payload.correctness_hint.get('error_message', 'Numerical stability problem')}\\n"
                if payload.correctness_hint.get("max_absolute_difference") != "N/A (from KernelBench)":
                    correctness_feedback_section += f"- Max Absolute Difference: {payload.correctness_hint.get('max_absolute_difference')}\\n"
                if payload.correctness_hint.get("relative_error") != "N/A (from KernelBench)":
                    correctness_feedback_section += f"- Max Relative Error: {payload.correctness_hint.get('relative_error')}\\n"
                if payload.correctness_hint.get("pytorch_output_preview") != "N/A (from KernelBench)":
                    correctness_feedback_section += f"- PyTorch Output Preview: {payload.correctness_hint.get('pytorch_output_preview')}\\n"
                if payload.correctness_hint.get("triton_output_preview") != "N/A (from KernelBench)":
                    correctness_feedback_section += f"- Triton Output Preview: {payload.correctness_hint.get('triton_output_preview')}\\n"
                correctness_feedback_section += "Please pay close attention to these numerical discrepancies and apply techniques to improve precision (e.g., higher-precision accumulators, operation reordering)."
                task_focus = "DEBUG and FIX the provided Triton kernel, focusing on NUMERICAL STABILITY. Ensure the overall kernel logic remains correct."
            
            else: # Other correctness issues
                critical_error_intro = "IMPORTANT: The kernel COMPILED successfully but FAILED correctness checks."
                correctness_feedback_section = f"\\n\\n--- Correctness Feedback ---\\n{json.dumps(payload.correctness_hint, indent=2)}"
                task_focus = "DEBUG and FIX the provided Triton kernel, focusing on the CORRECTNESS issues detailed above. Ensure the overall kernel logic remains correct."
        
        elif isinstance(payload.correctness_hint, str):
            # Handle string format (new enhanced feedback)
            correctness_hint_str = payload.correctness_hint
            
            # Determine criticality and type from string content
            if "CRITICAL GRID DIMENSIONALITY ERROR" in correctness_hint_str:
                critical_error_intro = (
                    "CRITICAL ISSUE: The kernel has a GRID DIMENSIONALITY ERROR. "
                    "The kernel uses tl.program_id(1) but is likely launched with a 1D grid."
                )
                task_focus = "DEBUG and FIX the grid dimensionality issue. Use proper 2D grid launch or modify kernel to use 1D grid."
            elif "POINTER ARITHMETIC ERROR" in correctness_hint_str:
                critical_error_intro = (
                    "CRITICAL ISSUE: The kernel has POINTER ARITHMETIC ERRORS. "
                    "Incorrect pointer calculations are causing memory access issues."
                )
                task_focus = "DEBUG and FIX the pointer arithmetic. Ensure proper stride multiplication and memory layout."
            elif "MASKING LOGIC ERROR" in correctness_hint_str:
                critical_error_intro = (
                    "CRITICAL ISSUE: The kernel has MASKING LOGIC ERRORS. "
                    "Incorrect boundary checking is causing out-of-bounds access."
                )
                task_focus = "DEBUG and FIX the masking logic. Use correct boundary checks and 2D masking patterns."
            elif "KERNELBENCH RUNTIME ERROR" in correctness_hint_str:
                critical_error_intro = (
                    "CRITICAL ISSUE: The kernel COMPILED successfully but FAILED during KernelBench validation "
                    "due to a runtime error."
                )
                task_focus = "DEBUG and FIX the runtime error. Focus on kernel structure and launcher function."
            else:
                critical_error_intro = "IMPORTANT: The kernel COMPILED successfully but FAILED correctness checks."
                task_focus = "DEBUG and FIX the provided Triton kernel based on the detailed feedback below."
            
            correctness_feedback_section = f"\\n\\n--- Detailed Correctness Feedback ---\\n{correctness_hint_str}"

    # --- Process Compiler Hint (error_hint) ---
    if payload.error_hint:
        # Check if correctness_hint contains 'kernelbench_runtime_error'
        # This handles correctness_hint being a string or a dictionary.
        is_kb_runtime_error_in_hint = False
        if isinstance(payload.correctness_hint, str) and "kernelbench_runtime_error" in payload.correctness_hint.lower():
            is_kb_runtime_error_in_hint = True
        elif isinstance(payload.correctness_hint, dict) and "kernelbench_runtime_error" in payload.correctness_hint.get("error_type", ""):
            is_kb_runtime_error_in_hint = True

        if is_kb_runtime_error_in_hint:
            compiler_hint_section = (
                f"\n\n--- Previous Compiler Hint (Likely Outdated) ---\n"
                f"The following compiler hint was also provided: {payload.error_hint}\n"
                f"However, since the kernel most recently COMPILED successfully and then failed with a RUNTIME error, "
                f"this compiler hint might be from an older attempt and is likely LESS RELEVANT than the runtime error described above. "
                f"Prioritize fixing the runtime error."
            )
        else:
            compiler_hint_section = f"\n\n--- Compiler Hint ---\n{payload.error_hint}"
            if not task_focus: # If correctness check passed or was not the primary issue
                 task_focus = "DEBUG and FIX the provided Triton kernel, focusing on resolving the COMPILER HINT. Ensure the overall kernel logic remains correct."


    research_section = f"\n\n--- Additional Research Context ---\n{payload.research_context}" if payload.research_context else ""

    # --- Performance hints section ---
    perf_hints_section = ""
    if payload.device_info:
        perf_hints_section += "\n\n--- Target Device Information (for performance optimization) ---\n"
        for key, value in payload.device_info.items():
            perf_hints_section += f"- {key}: {value}\n"
    
    if payload.memory_access_analysis:
        perf_hints_section += "\n--- Memory Access Analysis (for inputs) ---\n"
        for input_name, analysis_dict in payload.memory_access_analysis.items():
            perf_hints_section += f"- For {input_name}:\n"
            if isinstance(analysis_dict, dict):
                for key, value in analysis_dict.items():
                    if key == "recommendations" and isinstance(value, list):
                        if value:
                            perf_hints_section += f"  - {key.capitalize()}:\n"
                            for rec_item in value:
                                perf_hints_section += f"    - {rec_item}\n"
                    else:
                        perf_hints_section += f"  - {key.capitalize()}: {value}\n"
            else:
                 perf_hints_section += f"  - {analysis_dict}\n"
    
    if payload.profiling_hotspots:
        perf_hints_section += "\n--- PyTorch Profiling Hotspots (Performance Context) ---\n"
        
        # Extract metadata if available
        metadata = None
        hotspots_data = payload.profiling_hotspots
        if hotspots_data and hotspots_data[0].get("op") == "__PROFILING_METADATA__":
            metadata = hotspots_data[0].get("metadata", {})
            hotspots_data = hotspots_data[1:]  # Skip metadata entry
        
        if metadata:
            perf_hints_section += f"📊 PROFILING OVERVIEW:\n"
            perf_hints_section += f"  • Total operations profiled: {metadata.get('total_operations', 'unknown')}\n"
            perf_hints_section += f"  • Profiling iterations: {metadata.get('iterations', 'unknown')}\n"
            perf_hints_section += f"  • Input tensors: {metadata.get('input_tensor_count', 'unknown')} tensors\n"
            if metadata.get('input_tensor_shapes'):
                perf_hints_section += f"  • Input shapes: {metadata['input_tensor_shapes']}\n"
            if metadata.get('model_parameters', 0) > 0:
                perf_hints_section += f"  • Model parameters: {metadata['model_parameters']:,}\n"
            perf_hints_section += f"  • CUDA timing available: {'Yes' if metadata.get('has_cuda_timing') else 'No'}\n\n"
        
        perf_hints_section += "The following operations were identified as performance hotspots in the PyTorch baseline:\n"
        
        # Categorize operations for better optimization guidance
        compute_ops = []
        memory_ops = []
        elementwise_ops = []
        other_ops = []
        
        for hotspot in hotspots_data[:10]:  # Analyze top 10 hotspots
            category = hotspot.get("operation_category", "unknown")
            
            if category == "compute_intensive":
                compute_ops.append(hotspot)
            elif category == "memory_intensive":
                memory_ops.append(hotspot)
            elif category == "elementwise":
                elementwise_ops.append(hotspot)
            else:
                other_ops.append(hotspot)
        
        # Display compute-intensive operations
        if compute_ops:
            perf_hints_section += "\n🔥 COMPUTE-INTENSIVE OPERATIONS (Primary optimization targets):\n"
            for i, hotspot in enumerate(compute_ops[:5], 1):
                op_name = hotspot.get("op", "unknown")
                percent = hotspot.get("percent", 0.0)
                time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                time_value = hotspot.get(time_metric, 0.0)
                count = hotspot.get("count", 0)
                avg_time = hotspot.get("avg_time", 0.0)
                
                perf_hints_section += f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n"
                perf_hints_section += f"     Called {count} times, avg {avg_time:.1f}μs per call\n"
                
                # Add FLOPS information if available
                if hotspot.get("gflops_per_sec"):
                    perf_hints_section += f"     Performance: {hotspot['gflops_per_sec']:.1f} GFLOPS/sec\n"
                
                # Add memory information if available
                if hotspot.get("cuda_memory_usage"):
                    perf_hints_section += f"     CUDA memory: {hotspot['cuda_memory_usage']} bytes\n"
            
            perf_hints_section += "  → Focus on: Kernel fusion, tiling strategies, and memory coalescing\n"
            perf_hints_section += "  → Consider: Matrix multiplication optimizations, tensor core usage\n"
        
        # Display memory-intensive operations
        if memory_ops:
            perf_hints_section += "\n💾 MEMORY-INTENSIVE OPERATIONS (Secondary optimization targets):\n"
            for i, hotspot in enumerate(memory_ops[:3], 1):
                op_name = hotspot.get("op", "unknown")
                percent = hotspot.get("percent", 0.0)
                time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                time_value = hotspot.get(time_metric, 0.0)
                count = hotspot.get("count", 0)
                avg_time = hotspot.get("avg_time", 0.0)
                
                perf_hints_section += f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n"
                perf_hints_section += f"     Called {count} times, avg {avg_time:.1f}μs per call\n"
                
                if hotspot.get("cuda_memory_usage"):
                    perf_hints_section += f"     CUDA memory: {hotspot['cuda_memory_usage']} bytes\n"
            
            perf_hints_section += "  → Focus on: Reducing memory movements, vectorized loads/stores\n"
            perf_hints_section += "  → Consider: Memory layout optimizations, avoiding unnecessary copies\n"
        
        # Display elementwise operations
        if elementwise_ops:
            perf_hints_section += "\n⚡ ELEMENTWISE OPERATIONS (Fusion opportunities):\n"
            for i, hotspot in enumerate(elementwise_ops[:3], 1):
                op_name = hotspot.get("op", "unknown")
                percent = hotspot.get("percent", 0.0)
                time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                time_value = hotspot.get(time_metric, 0.0)
                
                perf_hints_section += f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n"
            
            perf_hints_section += "  → Focus on: Kernel fusion with compute operations\n"
            perf_hints_section += "  → Consider: Combining multiple elementwise ops into single kernel\n"
        
        # Display other operations
        if other_ops:
            perf_hints_section += "\n🔧 OTHER OPERATIONS:\n"
            for i, hotspot in enumerate(other_ops[:3], 1):
                op_name = hotspot.get("op", "unknown")
                percent = hotspot.get("percent", 0.0)
                time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                time_value = hotspot.get(time_metric, 0.0)
                
                perf_hints_section += f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n"
        
        # Calculate total hotspot coverage
        total_hotspot_percent = sum(h.get("percent", 0.0) for h in hotspots_data[:10])
        perf_hints_section += f"\n📈 PERFORMANCE ANALYSIS:\n"
        perf_hints_section += f"  • Top 10 operations account for {total_hotspot_percent:.1f}% of execution time\n"
        perf_hints_section += f"  • Total operations analyzed: {len(hotspots_data)}\n"
        
        # Provide optimization strategy based on operation mix
        perf_hints_section += "\n🎯 OPTIMIZATION STRATEGY:\n"
        if compute_ops and compute_ops[0].get("percent", 0.0) > 30:
            perf_hints_section += "  • PRIMARY: Optimize the compute-intensive operations above (>30% of time)\n"
            perf_hints_section += "  • Use kernel fusion to combine multiple operations into a single kernel\n"
            perf_hints_section += "  • Consider tiling and blocking strategies for large tensors\n"
            if any(h.get("gflops_per_sec", 0) > 0 for h in compute_ops):
                perf_hints_section += "  • Target operations with high FLOPS potential for maximum speedup\n"
        elif memory_ops and memory_ops[0].get("percent", 0.0) > 20:
            perf_hints_section += "  • PRIMARY: Optimize memory access patterns (>20% of time)\n"
            perf_hints_section += "  • Focus on memory coalescing and reducing memory bandwidth usage\n"
            perf_hints_section += "  • Consider eliminating unnecessary memory operations\n"
        else:
            perf_hints_section += "  • BALANCED: Optimize both compute and memory operations\n"
        
        if elementwise_ops:
            perf_hints_section += f"  • FUSION OPPORTUNITY: {len(elementwise_ops)} elementwise operations can be fused\n"
        
        perf_hints_section += "  • Target the highest-percentage operations first for maximum impact\n"
        perf_hints_section += "  • Consider the operation categories above when designing your kernel\n"

    # --- Construct the final prompt ---
    if payload.previous_kernel_src:
        if not task_focus: # Should have been set if there was any error
            task_focus = "DEBUG and FIX the provided Triton kernel. Review all provided feedback carefully."

        prompt = (
            f"{critical_error_intro}\n\n"
            f"The following PyTorch reference function was provided:\n```python\n{payload.pytorch_src}\n```\n"
            f"Input tensor specs (json):\n{specs_json}\n"
            f"The following Triton kernel was previously generated but requires fixing:\n```python\n{payload.previous_kernel_src}\n```\n"
            f"{correctness_feedback_section}{compiler_hint_section}{research_section}{perf_hints_section}\n\n"
            f"Your task: {task_focus}\n"
            f"Ensure the corrected kernel adheres to best practices (vectorization, num_warps/num_stages, boundary checks, memory coalescing). "
            f"If addressing numerical stability (especially with float16/bfloat16), prioritize solutions that enhance precision, such as using higher-precision accumulators (e.g., float32 for sums if inputs are float16), reordering operations, or applying stable mathematical identities, if these are not superseded by a critical runtime error. "
            f"Address ALL relevant feedback. If a specific runtime error is highlighted, that is your TOP PRIORITY. "
            f"Return ONLY the complete, valid, corrected Python code for the kernel and its launcher function, wrapped in a single markdown block."
        )
    else: # First attempt (no previous_kernel_src)
        task_focus = (
            "Write an efficient @triton.jit kernel based on the PyTorch reference, input specs, and any provided feedback (research, performance/device hints). "
            "Pay special attention to numerical stability if inputs are float16/bfloat16 (e.g., use float32 accumulators). "
            "Follow best practices: vectorization, correct num_warps/num_stages, boundary checks, and memory coalescing."
        )
        prompt = (
            f"PyTorch reference function:\n```python\n{payload.pytorch_src}\n```\n"
            f"Input tensor specs (json):\n{specs_json}{compiler_hint_section}{correctness_feedback_section}{research_section}{perf_hints_section}\n\n"
            f"Your task: {task_focus}\n"
            f"Return ONLY valid Python code containing the kernel plus a helper launcher function, wrapped in a single markdown block."
        )
    return prompt


synth_tool = FunctionTool(_synthesize_kernel)


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="synthesis",
            description="Generates Triton kernels using structured generation with LLM fallback.",
            tools=[synth_tool]
        )

    async def synthesize(self, payload: SynthIn) -> SynthOut:
        return _synthesize_kernel(payload)

    def _perform_direct_llm_synthesis(self, op_spec: Dict[str, Any], feedback: Optional[str] = None) -> str:
        """Perform kernel synthesis via direct LLM call with enhanced validation."""
        
        # Extract input specifications
        input_specs = op_spec.get("input_specs", [])
        pytorch_src = op_spec.get("pytorch_src", "")
        
        # Build synthesis prompt with error prevention guidelines
        prompt = self._build_synthesis_prompt(input_specs, pytorch_src, feedback)
        
        # Add specific error prevention guidelines
        error_prevention_guidelines = """
        
CRITICAL ERROR PREVENTION GUIDELINES:
1. SHAPE HANDLING: Always validate tensor shapes before indexing. Use len(shape) checks.
2. SAFE INDEXING: Use shape[-1] only after verifying len(shape) >= 1
3. TENSOR CREATION: Always specify device and dtype explicitly
4. MEMORY ACCESS: Use proper masking in tl.load/tl.store operations
5. GRID CALCULATION: Ensure grid dimensions match kernel expectations
6. PARAMETER VALIDATION: Check all input parameters before use
7. LAUNCHER FUNCTION: Always provide a launcher function that handles shape validation

Example safe shape handling:
```python
def launch_layer_norm(output, input_tensor, weight, bias, eps=1e-5):
    # Safe shape extraction
    if len(input_tensor.shape) < 2:
        raise ValueError(f"Input must have at least 2 dimensions, got {input_tensor.shape}")
    
    # For 3D input [batch, seq, dim], reshape to 2D [batch*seq, dim]
    original_shape = input_tensor.shape
    if len(original_shape) == 3:
        batch_size, seq_len, hidden_dim = original_shape
        input_2d = input_tensor.view(-1, hidden_dim)
        output_2d = output.view(-1, hidden_dim)
        total_rows = batch_size * seq_len
    elif len(original_shape) == 2:
        total_rows, hidden_dim = original_shape
        input_2d = input_tensor
        output_2d = output
    else:
        raise ValueError(f"Unsupported input shape: {original_shape}")
```
"""
        
        full_prompt = prompt + error_prevention_guidelines
        
        logger.debug(f"Prompt sent to LLM (via direct call): {full_prompt[:200]}...")
        
        # Generate kernel code
        try:
            response = self.llm_client.generate_response([
                {"role": "system", "content": "You are an expert at writing highly-optimized Triton kernels."},
                {"role": "user", "content": full_prompt}
            ])
            
            # Extract and validate the code
            kernel_code = self._extract_code_from_response(response)
            
            # Perform static validation before returning
            validation_result = self._validate_kernel_code_static(kernel_code)
            if not validation_result.is_valid:
                logger.warning(f"Generated kernel failed static validation: {validation_result.error_message}")
                # Add validation feedback and retry once
                validation_feedback = f"VALIDATION ERROR: {validation_result.error_message}. Please fix these issues."
                retry_prompt = full_prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {validation_feedback}"
                
                response = self.llm_client.generate_response([
                    {"role": "system", "content": "You are an expert at writing highly-optimized Triton kernels."},
                    {"role": "user", "content": retry_prompt}
                ])
                kernel_code = self._extract_code_from_response(response)
            
            logger.debug(f"Raw LLM output received (len={len(kernel_code)}, direct call):\n{kernel_code[:500]}...")
            return kernel_code
            
        except Exception as e:
            logger.error(f"Direct LLM synthesis failed: {str(e)}")
            raise RuntimeError(f"LLM synthesis error: {str(e)}")

    def _validate_kernel_code_static(self, kernel_code: str) -> ValidationResult:
        """Perform static validation of generated kernel code to catch common errors."""
        
        try:
            # Check for common error patterns
            errors = []
            warnings = []
            
            # 1. Check for unsafe shape indexing
            if ".shape[-2]" in kernel_code and "len(" not in kernel_code:
                errors.append("Unsafe shape indexing: .shape[-2] used without length check")
            
            # 2. Check for missing launcher function
            if "def launch_" not in kernel_code and "def " in kernel_code:
                if "@triton.jit" in kernel_code:
                    errors.append("Missing launcher function: Kernel defined but no launch_ function provided")
            
            # 3. Check for tensor creation without device specification
            if "torch.empty(" in kernel_code and "device=" not in kernel_code:
                warnings.append("Tensor creation without explicit device specification")
            
            # 4. Check for proper grid calculation
            if "grid = " in kernel_code:
                if "triton.cdiv" not in kernel_code and "total_rows" not in kernel_code:
                    warnings.append("Grid calculation may not handle non-divisible sizes properly")
            
            # 5. Check for proper masking in memory operations
            if "tl.load(" in kernel_code and "mask=" not in kernel_code:
                warnings.append("Memory load operations without masking may cause out-of-bounds access")
            
            # 6. Check for shape validation in launcher
            if "def launch_" in kernel_code:
                launcher_section = kernel_code[kernel_code.find("def launch_"):]
                if "len(" not in launcher_section and ".shape" in launcher_section:
                    errors.append("Launcher function accesses .shape without length validation")
            
            # 7. Try to compile the code to catch syntax errors
            try:
                compile(kernel_code, '<string>', 'exec')
            except SyntaxError as e:
                errors.append(f"Syntax error: {str(e)}")
            
            if errors:
                return ValidationResult(False, "; ".join(errors), warnings)
            else:
                return ValidationResult(True, "", warnings)
                
        except Exception as e:
            return ValidationResult(False, f"Validation failed: {str(e)}")
