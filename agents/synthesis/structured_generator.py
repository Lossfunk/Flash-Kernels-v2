"""
Simplified Triton Kernel Generator

This module provides a direct LLM-based approach to generating Triton kernels
from PyTorch reference code, eliminating complex AST parsing.
"""

import torch
import triton
import triton.language as tl
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agents.contracts import TensorSpec, SynthIn, SynthOut
from utils.logging_utils import get_logger

# Import the Triton documentation knowledge base
from agents.synthesis.triton_docs_knowledge import triton_docs_kb

logger = get_logger("StructuredKernelGenerator")


@dataclass
class PromptContext:
    """Context for LLM prompt generation."""
    pytorch_src: str
    input_specs: List[TensorSpec]
    device_info: Optional[Dict[str, Any]] = None
    memory_access_analysis: Optional[Dict[str, Any]] = None
    research_context: Optional[str] = None
    error_hint: Optional[str] = None
    previous_kernel_src: Optional[str] = None
    profiling_hotspots: Optional[List[Dict[str, Any]]] = None


class StructuredKernelGenerator:
    """Main class for simplified kernel generation using direct LLM approach."""
    
    def __init__(self):
        pass
    
    @classmethod
    def can_use_research_context(cls) -> bool:
        """Indicate that this generator can process research context."""
        return True
    
    def generate(self, payload: SynthIn) -> SynthOut:
        """Generate a Triton kernel using direct LLM synthesis."""
        logger.info("Starting simplified kernel generation (direct LLM approach)")
        
        device_info = payload.device_info
        memory_analysis = payload.memory_access_analysis
        if device_info:
            logger.info("Device info provided: %s", device_info.get("device_name", "Unknown"))
        if memory_analysis:
            logger.info("Memory analysis provided for %d inputs", len(memory_analysis))

        try:
            # First, check if we have an optimized implementation from Triton docs
            # Try problem ID detection first, then fall back to PyTorch source analysis
            problem_id = payload.problem_id
            detected_operation = triton_docs_kb.detect_operation_from_problem_id(problem_id, payload.pytorch_src)
            
            if detected_operation and triton_docs_kb.has_implementation(detected_operation):
                logger.info(f"🎯 TRITON DOCS MATCH: Detected '{detected_operation}' operation")
                if problem_id is not None:
                    logger.info(f"   Matched via problem ID {problem_id}")
                else:
                    logger.info("   Matched via PyTorch source analysis")
                logger.info("Using optimized implementation from official Triton documentation")
                
                optimized_kernel = triton_docs_kb.generate_optimized_kernel(
                    detected_operation, 
                    payload.input_specs, 
                    device_info
                )
                
                if optimized_kernel:
                    logger.info("Successfully generated kernel from Triton docs knowledge base")
                    
                    # Basic syntax sanity-check
                    if self._syntax_check(optimized_kernel):
                        logger.info("✅ Triton docs kernel passed syntax check")
                        return SynthOut(kernel_src=optimized_kernel)
                    else:
                        logger.warning("⚠️ Triton docs kernel failed syntax check, falling back to LLM")
                        # Fall through to LLM generation
                else:
                    logger.warning("Failed to generate kernel from Triton docs, falling back to LLM")
                    # Fall through to LLM generation
            else:
                if detected_operation:
                    logger.info(f"Detected operation '{detected_operation}' but no optimized implementation available")
                else:
                    logger.info("Could not detect operation type from problem ID or PyTorch source")
                logger.info("Proceeding with LLM-based generation")

            # Create prompt context for LLM fallback
            prompt_ctx = PromptContext(
                pytorch_src=payload.pytorch_src,
                input_specs=payload.input_specs,
                device_info=device_info,
                memory_access_analysis=memory_analysis,
                research_context=payload.research_context,
                error_hint=payload.error_hint,
                previous_kernel_src=payload.previous_kernel_src,
                profiling_hotspots=payload.profiling_hotspots
            )

            logger.info("Delegating kernel generation to LLM")
            kernel_src = self._call_llm(prompt_ctx, payload)

            # Basic syntax sanity-check
            if not self._syntax_check(kernel_src):
                logger.warning("LLM-generated kernel failed syntax check")
                logger.warning("Problematic kernel_src (first 2000 chars):\n%s", kernel_src[:2000])
                
                # Return the failed kernel anyway - let the pipeline handle it
                # This allows the error to be captured and used for research/feedback
                logger.info("Returning failed kernel for pipeline error handling")
                return SynthOut(kernel_src=kernel_src)

            logger.info("Returning LLM-generated kernel (passed syntax check)")
            return SynthOut(kernel_src=kernel_src)

        except ImportError as e: # Should ideally be caught in _call_llm, but good to have
            logger.error("ImportError during structured generation: %s", e, exc_info=True)
            raise RuntimeError(f"ImportError in StructuredKernelGenerator: {str(e)}") from e
        except Exception as e:
            # For any other unexpected errors specifically within this generator's logic
            logger.error("StructuredKernelGenerator encountered an internal unexpected error: %s", e, exc_info=True)
            # Propagate as a generic runtime error, rather than using the basic template.
            # This allows the main agent to decide on the fallback strategy.
            raise RuntimeError(f"Internal error in StructuredKernelGenerator: {str(e)}") from e
    
    def _build_llm_prompt(self, ctx: PromptContext) -> str:
        """Build a comprehensive prompt for the LLM."""
        lines = []
        
        # Header
        lines.append("You are an expert at writing highly-optimized Triton kernels.\n")
        
        # Check if we have relevant Triton docs knowledge to include
        detected_operation = triton_docs_kb.detect_operation_from_pytorch(ctx.pytorch_src)
        if detected_operation and triton_docs_kb.has_implementation(detected_operation):
            impl = triton_docs_kb.get_implementation(detected_operation)
            lines.append(f"🎯 REFERENCE IMPLEMENTATION AVAILABLE:\n")
            lines.append(f"For {detected_operation} operations, refer to the official Triton documentation:\n")
            lines.append(f"Source: {impl.source_url}\n")
            lines.append(f"Description: {impl.description}\n")
            lines.append(f"Performance notes: {impl.performance_notes}\n")
            lines.append("Use this as inspiration for your implementation.\n\n")
        
        # Input specifications
        lines.append("Input tensors:\n")
        for i, spec in enumerate(ctx.input_specs):
            lines.append(f"  - input{i}: shape={spec.shape}, dtype={spec.dtype}\n")
        
        # Device information
        if ctx.device_info:
            lines.append(f"Target device: {ctx.device_info.get('device_name', 'Unknown')}\n")
            if 'compute_capability' in ctx.device_info:
                lines.append(f"Compute capability: {ctx.device_info['compute_capability']}\n")
        
        # Memory analysis hints
        if ctx.memory_access_analysis:
            lines.append("Memory access patterns analyzed - optimize for memory coalescing.\n")
        
        # Profiling hotspots context
        if ctx.profiling_hotspots:
            lines.append("PyTorch Profiling Hotspots (Performance Context):\n")
            lines.append("The following operations were identified as performance bottlenecks in the PyTorch baseline:\n")
            
            # Categorize operations for better optimization guidance
            compute_ops = []
            memory_ops = []
            other_ops = []
            
            for hotspot in ctx.profiling_hotspots[:10]:  # Analyze top 10 hotspots
                op_name = hotspot.get("op", "unknown").lower()
                
                # Categorize operations
                if any(compute_keyword in op_name for compute_keyword in 
                       ["mm", "matmul", "conv", "gemm", "dot", "addmm", "bmm", "baddbmm"]):
                    compute_ops.append(hotspot)
                elif any(memory_keyword in op_name for memory_keyword in 
                         ["copy", "transpose", "permute", "view", "reshape", "contiguous", "cat", "stack"]):
                    memory_ops.append(hotspot)
                else:
                    other_ops.append(hotspot)
            
            # Display compute-intensive operations
            if compute_ops:
                lines.append("🔥 COMPUTE-INTENSIVE OPERATIONS (Primary optimization targets):\n")
                for i, hotspot in enumerate(compute_ops[:5], 1):
                    op_name = hotspot.get("op", "unknown")
                    percent = hotspot.get("percent", 0.0)
                    time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                    time_value = hotspot.get(time_metric, 0.0)
                    lines.append(f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n")
                lines.append("  → Focus on: Kernel fusion, tiling strategies, and memory coalescing\n")
            
            # Display memory-intensive operations
            if memory_ops:
                lines.append("💾 MEMORY-INTENSIVE OPERATIONS (Secondary optimization targets):\n")
                for i, hotspot in enumerate(memory_ops[:3], 1):
                    op_name = hotspot.get("op", "unknown")
                    percent = hotspot.get("percent", 0.0)
                    time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                    time_value = hotspot.get(time_metric, 0.0)
                    lines.append(f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n")
                lines.append("  → Focus on: Reducing memory movements, vectorized loads/stores\n")
            
            # Display other operations
            if other_ops:
                lines.append("⚡ OTHER OPERATIONS:\n")
                for i, hotspot in enumerate(other_ops[:3], 1):
                    op_name = hotspot.get("op", "unknown")
                    percent = hotspot.get("percent", 0.0)
                    time_metric = "self_cuda_time_total" if "self_cuda_time_total" in hotspot else "self_cpu_time_total"
                    time_value = hotspot.get(time_metric, 0.0)
                    lines.append(f"  {i}. {op_name}: {percent}% of total time ({time_value:.1f}μs)\n")
            
            # Calculate total hotspot coverage
            total_hotspot_percent = sum(h.get("percent", 0.0) for h in ctx.profiling_hotspots[:10])
            lines.append(f"📊 PROFILING SUMMARY:\n")
            lines.append(f"  • Top operations account for {total_hotspot_percent:.1f}% of execution time\n")
            lines.append(f"  • Total operations profiled: {len(ctx.profiling_hotspots)}\n")
            
            # Provide optimization strategy
            lines.append("🎯 OPTIMIZATION STRATEGY:\n")
            if compute_ops and compute_ops[0].get("percent", 0.0) > 30:
                lines.append("  • PRIMARY: Optimize the compute-intensive operations above (>30% of time)\n")
                lines.append("  • Use kernel fusion to combine multiple operations into a single kernel\n")
                lines.append("  • Consider tiling and blocking strategies for large tensors\n")
            elif memory_ops and memory_ops[0].get("percent", 0.0) > 20:
                lines.append("  • PRIMARY: Optimize memory access patterns (>20% of time)\n")
                lines.append("  • Focus on memory coalescing and reducing memory bandwidth usage\n")
            else:
                lines.append("  • BALANCED: Optimize both compute and memory operations\n")
            
            lines.append("  • Target the highest-percentage operations first for maximum impact\n")
            lines.append("  • Consider the operation categories above when designing your kernel\n")
            lines.append("Focus your Triton kernel optimization on the most time-consuming operations listed above.\n")
        
        # Previous attempt context
        if ctx.error_hint:
            lines.append(f"Previous attempt failed with error: {ctx.error_hint}\n")
            lines.append("Please fix these issues in your implementation.\n")
        
        if ctx.previous_kernel_src:
            lines.append("Previous kernel attempt (for reference):\n")
            lines.append("```python\n")
            lines.append(ctx.previous_kernel_src[:1000])  # Limit size
            if len(ctx.previous_kernel_src) > 1000:
                lines.append("\n... (truncated)")
            lines.append("\n```\n")
        
        # Research context
        if ctx.research_context:
            lines.append("Research insights:\n")
            lines.append(ctx.research_context)
            lines.append("\n")
        
        # PyTorch reference code
        lines.append("PyTorch reference implementation:\n")
        lines.append("```python\n")
        lines.append(ctx.pytorch_src.strip())
        lines.append("\n```\n")
        
        # Instructions
        lines.append("\nPlease generate a complete, optimized Triton kernel that implements the same functionality as the PyTorch reference code.\n")
        lines.append("\nRequirements:\n")
        lines.append("1. Include all necessary imports (torch, triton, triton.language as tl)\n")
        lines.append("2. Write the @triton.jit kernel function\n")
        lines.append("3. Write a launcher function that MUST be named starting with 'launch_' (e.g., 'launch_relu', 'launch_matmul')\n")
        lines.append("4. The launcher function should handle tensor creation, grid calculation, and kernel invocation\n")
        lines.append("5. Handle memory coalescing and boundary conditions properly\n")
        lines.append("6. Use appropriate block sizes for the target device\n")
        lines.append("7. Ensure numerical correctness\n")
        lines.append("8. Triton reduction ops do NOT support 'keepdim'; instead reduce and then broadcast the result manually if needed.\n")
        lines.append("9. For `tl.load`, use valid eviction policies. Avoid using 'model_prefer_reading'. If unsure, you can omit the `eviction_policy` parameter to use the default, or use common policies like 'evict_first' or 'evict_last'. Prioritize using features and parameters explicitly documented in the official Triton documentation.\n")
        
        # Add specific guidance for known operations
        if detected_operation:
            lines.append(f"10. This appears to be a {detected_operation} operation. Consider the following optimizations:\n")
            if detected_operation == "softmax":
                lines.append("    - Use kernel fusion to minimize memory I/O\n")
                lines.append("    - Process row-wise for numerical stability\n")
                lines.append("    - Use power-of-2 block sizes\n")
                lines.append("    - Consider occupancy-based grid sizing\n")
        
        lines.append("\nOutput ONLY the complete Python code wrapped in a single markdown block:\n")
        lines.append("```python\n# Your complete Triton kernel implementation here\n```\n")
        lines.append("\nDo NOT include any other text or commentary outside the markdown block.")
        
        return "".join(lines)

    def _call_llm(self, prompt_ctx: PromptContext, payload: SynthIn) -> str:
        """Call the LLM with the constructed prompt."""
        try:
            from copy import deepcopy
            from agents.synthesis.agent import _perform_direct_llm_synthesis
        except ImportError as e:
            logger.error("LLM synthesis path unavailable: %s", e)
            raise

        # Create enriched payload with our comprehensive prompt
        enriched_payload = deepcopy(payload)
        enriched_payload.research_context = self._build_llm_prompt(prompt_ctx)
        
        synth_out = _perform_direct_llm_synthesis(enriched_payload)
        return synth_out.kernel_src

    def _fallback_to_basic_template(self, payload: SynthIn) -> SynthOut:
        """Provide a basic template as ultimate fallback."""
        logger.warning("Using basic template fallback")
        
        # Create a simple elementwise template as fallback
        basic_kernel = '''
import torch
import triton
import triton.language as tl

@triton.jit
def basic_kernel(output_ptr, input0_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input0_ptr + offsets, mask=mask)
    
    # Simple passthrough (modify as needed)
    result = x
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def launch_basic_kernel(output, input0):
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    basic_kernel[grid](output, input0, n_elements, BLOCK_SIZE=256)
'''
        
        return SynthOut(kernel_src=basic_kernel)

    @staticmethod
    def _syntax_check(code: str) -> bool:
        """Return True if the given source compiles syntactically."""
        try:
            compile(code, "<triton_kernel_src>", "exec")
            return True
        except SyntaxError:
            return False 