import asyncio
import uuid
import hashlib
import json
import datetime
import torch
import random
import re
import os
import types
import textwrap
from collections import Counter
from utils.logging_utils import get_logger
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from agents.orchestrator.agent import OrchestratorAgent
from agents.memory.agent import MemoryAgent
from agents.synthesis.agent import SynthesisAgent
from agents.compile.agent import CompileAgent
from agents.reasoner.agent import ReasonerAgent
from agents.correctness.agent import CorrectnessAgent
from agents.fallback.agent import FallbackAgent
from agents.correctness_reasoner.agent import CorrectnessReasonerAgent
from agents.correctness_reasoner_memory.agent import CorrectnessReasonerMemoryAgent
from agents.observers import SQLiteObserver
from misc.deep_research_agent import DeepResearchManager
from agents.performance.agent import PerformanceAgent
from agents.performance.proton_analyzer import ProtonRooflineAnalyzer, MemoryAccessOptimizer

from agents.contracts import (
    OrchestratorIn,
    MemoryQueryIn,
    MemoryPutIn,
    SynthIn,
    CompileIn,
    CompileOut,
    ReasonerIn,
    CorrectIn,
    CorrectOut,
    FallbackIn,
    CorrectnessReasonerIn,
    CorrectnessReasonerMemoryIn,
    CorrectnessReasonerMemoryKey,
    CorrectnessReasonerMemoryAddIn,
    CorrectnessReasonerMemoryGetIn,
    CorrectnessReasonerMemoryClearIn,
    ReasoningAttemptDetail,
    TensorSpec,
    PerformanceIn,
    PerformanceOut
)

logger = get_logger("KernelPipelineAgent")

class KernelPipelineAgent:
    """Manual workflow orchestration to match KernelBench message flow.
    
    Configuration Examples:
    
    # Disable early termination completely (let it run all attempts)
    pipeline = KernelPipelineAgent(enable_early_termination=False)
    
    # More aggressive early termination (terminate faster)
    pipeline = KernelPipelineAgent(
        consecutive_error_threshold=2,  # Terminate after 2 consecutive identical errors
        min_attempts_before_termination=1,  # Allow termination after just 1 attempt
        dominance_threshold=0.6  # Terminate if 60% of errors are the same type
    )
    
    # More lenient early termination (current default)
    pipeline = KernelPipelineAgent(
        consecutive_error_threshold=4,  # Terminate after 4 consecutive identical errors
        min_attempts_before_termination=3,  # Require at least 3 attempts
        dominance_threshold=0.8  # Terminate if 80% of errors are the same type
    )
    
    # Disable diversity injection (use original approach only)
    pipeline = KernelPipelineAgent(enable_diversity_injection=False)
    
    # More aggressive diversity injection
    pipeline = KernelPipelineAgent(diversity_trigger_threshold=1)  # Inject after 1 repeated error
    
    # Force maximum exploration
    pipeline = KernelPipelineAgent(
        max_synthesis_attempts=20,  # Try more synthesis attempts
        max_correctness_attempts=10,  # Try more parameter reasoning per synthesis
        enable_early_termination=False,  # Never terminate early
        diversity_trigger_threshold=1,  # Always inject diversity hints
        consecutive_errors_before_research=2 # Example for new param
    )
    """

    def __init__(self, 
                 max_synthesis_attempts: int = 20, 
                 max_correctness_attempts: int = 5,
                 # Early termination configuration
                 enable_early_termination: bool = False,  # Temporarily disabled to allow more attempts
                 min_attempts_before_termination: int = 3,
                 consecutive_error_threshold: int = 10,
                 alternating_pattern_threshold: int = 6,
                 dominance_threshold: float = 0.8,
                 # Diversity injection configuration
                 enable_diversity_injection: bool = True,  # Enable to break error patterns
                 diversity_trigger_threshold: int = 2,  # Trigger after 2 repeated errors
                 # Deep research configuration
                 consecutive_errors_before_research: int = 2,
                 web_search_tool_func=None,
                 # Performance tuning
                 target_speedup_threshold: float = 1.2,
                 min_efficiency_for_no_hint: float = 50.0,
                 enable_deep_research: bool = True
                 ):
        self.name = "kernel_pipeline"  # for logging
        self.logger = logger
        self.max_synthesis_attempts = max_synthesis_attempts
        self.max_parameter_reasoning_attempts = max_correctness_attempts
        
        # Early termination settings
        self.enable_early_termination = enable_early_termination
        self.min_attempts_before_termination = min_attempts_before_termination
        self.consecutive_error_threshold = consecutive_error_threshold
        self.alternating_pattern_threshold = alternating_pattern_threshold
        self.dominance_threshold = dominance_threshold
        
        # Diversity injection settings
        self.enable_diversity_injection = enable_diversity_injection
        self.diversity_trigger_threshold = diversity_trigger_threshold
        
        # Deep research toggle must be set BEFORE any conditional logic that references it
        self.enable_deep_research = enable_deep_research
        
        self.observer = SQLiteObserver()
        
        # Early termination tracking - Made more lenient
        self.synthesis_error_patterns: List[str] = []  # Track synthesis error patterns
        self.correctness_error_patterns: List[str] = []  # Track correctness error patterns
        self.consecutive_identical_errors = 5  # Increased from 3 to 5
        self.max_identical_errors = 7
        self.min_attempts_before_termination = min_attempts_before_termination  # Minimum attempts before considering termination
        
        # Diversity injection settings
        self.diversity_triggers: Dict[str, List[str]] = {
            "missing_arg": ["Try a completely different kernel structure", 
                          "Use different variable names and parameter ordering",
                          "Implement using a different algorithmic approach"],
            "shape_mismatch": ["Reconsider the output tensor dimensions",
                             "Try different block/grid size calculations", 
                             "Use alternative memory layout patterns"],
            "value_mismatch": ["Try a different mathematical approach",
                             "Use different precision or numerical methods",
                             "Implement with alternative loop structures"],
            "compile_error": ["Try simpler kernel structure first",
                            "Use different Triton language features",
                            "Implement with basic operations and build up"],
            "kernelbench_grid_dimensionality": [
                "Use 1D grid with single program_id(0) and calculate both M and K indices from it",
                "Implement proper 2D grid launch with grid = (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))",
                "Try element-wise kernel approach instead of tiled 2D approach"
            ],
            "kernelbench_pointer_arithmetic": [
                "Use simple linear indexing: ptr + row * stride + col instead of complex offset calculations",
                "Implement pointer arithmetic step-by-step with intermediate variables for clarity",
                "Try different memory access patterns like row-major vs column-major"
            ],
            "kernelbench_masking": [
                "Use simple boundary checks: idx < size instead of complex mask calculations",
                "Implement masking with explicit if-conditions rather than mask tensors",
                "Try different approaches to handle boundary conditions"
            ],
            "kernelbench_runtime": [
                "Simplify the kernel to basic element-wise operations first",
                "Use different grid/block size combinations",
                "Try alternative Triton programming patterns"
            ]
        }
        self.diversity_attempt_threshold = 2  # After 2 identical errors, inject diversity
        
        # Deep Research settings
        self.CONSECUTIVE_ERRORS_BEFORE_RESEARCH = consecutive_errors_before_research
        
        # Store web search function for deep research
        self.web_search_tool_func = web_search_tool_func
        
        # Research context cache - maps error signature hash to research context
        self.research_context_cache: Dict[str, str] = {}
        
        # Performance feedback
        self.last_performance_feedback: Optional[str] = None
        self.target_speedup_threshold = target_speedup_threshold
        self.min_efficiency_for_no_hint = min_efficiency_for_no_hint
        
        # Instantiate all agents
        self.orchestrator = OrchestratorAgent()
        self.memory = MemoryAgent()
        self.synthesis_agent = SynthesisAgent()
        self.compile_agent = CompileAgent()
        self.compile_reasoner_agent = ReasonerAgent()
        self.correctness_agent = CorrectnessAgent()
        self.fallback_agent = FallbackAgent()
        self.correctness_reasoner = CorrectnessReasonerAgent()
        self.correctness_reasoner_memory = CorrectnessReasonerMemoryAgent()
        self.deep_research_manager = None
        if self.enable_deep_research:
            self.deep_research_manager = DeepResearchManager(web_search_func=web_search_tool_func)
        self.performance_agent = PerformanceAgent()

    def _extract_error_signature(self, error_details: dict) -> str:
        """Extract a signature from error details to detect patterns."""
        if not error_details:
            return "unknown_error"
        
        error_type = error_details.get("error_type", "unknown")
        error_message_raw = error_details.get("error_message", "")
        error_message = str(error_message_raw) if error_message_raw is not None else ""

        # Check for CUDA OOM
        if "cuda out of memory" in error_message.lower() or \
           "could not allocate memory on cuda" in error_message.lower() or \
           "cublas_status_alloc_failed" in error_message.lower() or \
           "curand_status_allocation_failed" in error_message.lower():
            return "cuda_oom_error"
        
        # Enhanced KernelBench runtime error detection
        if error_type == "kernelbench_runtime_error":
            # Check for specific Triton runtime patterns
            if "tl.program_id(1)" in error_message and "pid_k" in error_message:
                return "kernelbench_grid_dimensionality_error"
            elif "input_row_start" in error_message and "output_row_start" in error_message:
                return "kernelbench_pointer_arithmetic_error"
            elif "mask_m" in error_message and "mask_k" in error_message:
                return "kernelbench_masking_logic_error"
            elif "tl.load" in error_message and "offsets" in error_message:
                return "kernelbench_memory_access_error"
            elif "at " in error_message and ":" in error_message:
                # Extract line number for more specific error tracking
                import re
                line_match = re.search(r"at (\d+):", error_message)
                if line_match:
                    return f"kernelbench_runtime_line_{line_match.group(1)}"
            return "kernelbench_runtime_error_generic"
        
        if error_type == "triton_execution_error":
            if "missing a required argument" in error_message:
                import re
                match = re.search(r"missing a required argument: '([^']+)'", error_message)
                if match:
                    return f"missing_arg_{match.group(1)}"
            elif "Error evaluating suggested_grid_str" in error_message: # Correctness agent
                return "invalid_grid_syntax_from_correctness"
            elif "invalid syntax" in error_message.lower() and ("<unknown>, line" in error_message or ".kernel.py" in error_message): # Compile error
                 return "compile_syntax_error"
            elif "expected a '.name' or '.value'" in error_message: # Compile error related to tl.constexpr
                 return "compile_constexpr_expected_name_or_value"
            elif "tl.constexpr" in error_message and "must be compile time constant" in error_message:
                 return "compile_constexpr_not_constant"
            elif "invalid type for argument" in error_message and "tl.constexpr" in error_message: # Compile error
                 return "compile_constexpr_invalid_type"
            elif "cannot be used as a tl.constexpr" in error_message: # Compile error
                 return "compile_cannot_be_constexpr"
            elif "expression cannot be used as a specialization constant" in error_message:
                 return "compile_expr_not_specialization_constant"
            elif "Couldn't find definition for operator" in error_message: # Triton compile error
                 return f"compile_op_not_found_{error_message.split(':')[-1].strip()}"
            elif "All kernel execution attempts failed" in error_message: # Correctness agent
                return "all_execution_attempts_failed_from_correctness"
            # Add more specific triton execution error parsers if needed
            return "triton_execution_error_generic"
        elif error_type == "shape_mismatch":
            expected_shape = error_details.get("expected_shape", [])
            actual_shape = error_details.get("actual_shape", [])
            return f"shape_mismatch_{len(expected_shape)}dims_vs_{len(actual_shape)}dims"
        elif error_type == "value_mismatch":
            return "value_mismatch"
        elif error_type == "compile_error": # Generic compile error from CompileAgent
            # Try to get more specific info from the log
            log_lower = error_message.lower()
            if "invalid syntax" in log_lower: return "compile_syntax_error_generic"
            if "undefined name" in log_lower: return "compile_undefined_name"
            if "type mismatch" in log_lower: return "compile_type_mismatch"
            return "compile_error_generic"
        
        # Fallback: use a hash of the error message for very specific, unparsed errors
        # Limit length to keep it manageable
        if error_message and len(error_message) > 20: # Only hash if somewhat substantial
            short_error_hash = hashlib.md5(error_message[:200].encode()).hexdigest()[:8]
            return f"{error_type}_{short_error_hash}"

        return f"{error_type}_generic" # Fallback for other types or very short messages

    def _should_terminate_early(self, current_error_signature: str, attempt_type: str) -> bool:
        """Check if we should terminate early due to repeated error patterns."""
        if not self.enable_early_termination: return False
            
        patterns = self.synthesis_error_patterns if "synthesis" in attempt_type else self.correctness_error_patterns
        
        if len(patterns) < self.min_attempts_before_termination: return False
        
        consecutive_count = 0
        for i in range(len(patterns) - 1, -1, -1):
            if patterns[i] == current_error_signature: consecutive_count += 1
            else: break
        
        if consecutive_count >= self.consecutive_error_threshold:
            self.logger.warning(f"Early termination: Identical {attempt_type} error '{current_error_signature}' repeated {consecutive_count} times (threshold: {self.consecutive_error_threshold})")
            return True
        
        if len(patterns) >= self.alternating_pattern_threshold:
            # Check for A-B-A-B... of length alternating_pattern_threshold
            # Example: if threshold is 6, check for A-B-A-B-A-B
            # The pattern is (X, Y) repeated N times, where 2*N = threshold
            # So, check patterns[-(threshold):]
            sub_pattern = patterns[-(self.alternating_pattern_threshold):]
            if len(set(sub_pattern)) == 2: # Ensure only two unique errors
                a, b = sub_pattern[0], sub_pattern[1]
                if a == b: return False # Not alternating if a and b are same
                alternating = True
                for i in range(self.alternating_pattern_threshold):
                    if (i % 2 == 0 and sub_pattern[i] != a) or \
                       (i % 2 == 1 and sub_pattern[i] != b):
                        alternating = False; break
                if alternating:
                    self.logger.warning(f"Early termination: Alternating {attempt_type} error pattern '{a}' <-> '{b}' detected for {self.alternating_pattern_threshold} attempts.")
                    return True
        
        if len(patterns) >= 5: # Min patterns for dominance check
            from collections import Counter
            pattern_counts = Counter(patterns)
            max_count = 0
            dominant_pattern = None
            for pattern, count in pattern_counts.items():
                if count > max_count: max_count = count; dominant_pattern = pattern
            
            if dominant_pattern and (max_count / len(patterns)) >= self.dominance_threshold:
                self.logger.warning(f"Early termination: {attempt_type} error pattern '{dominant_pattern}' dominates {max_count}/{len(patterns)} attempts ({(max_count/len(patterns))*100:.1f}%, threshold: {self.dominance_threshold*100:.1f}%).")
                return True
        return False

    def _log_attempt_summary(self, attempt_num: int, total_attempts: int, attempt_type: str, 
                           success: bool, error_signature: Optional[str] = None):
        status_char = "✓" if success else "✗"
        log_level = self.logger.info if success else self.logger.warning
        
        log_level("%s %s attempt %d/%d: %s%s", 
                  status_char, attempt_type.capitalize(), attempt_num, total_attempts,
                  "SUCCESS" if success else "FAILED",
                  f" - Error pattern: {error_signature}" if not success and error_signature else "")
            
        patterns_to_show = self.synthesis_error_patterns if "synthesis" in attempt_type else self.correctness_error_patterns
        if len(patterns_to_show) > 1:
            # Show last 3-5 patterns for context
            history_limit = min(len(patterns_to_show), 5) 
            recent_patterns = patterns_to_show[-history_limit:]
            self.logger.info("Recent %s error patterns: %s", attempt_type, " -> ".join(recent_patterns))

    def _inject_diversity_hint(self, current_error_signature: str, attempt_type_key: str, 
                               attempt_number_for_log: int) -> Optional[str]:
        """Inject diversity hints to break error patterns."""
        if not self.enable_diversity_injection: 
            return None
       
        # Determine which pattern list to check based on attempt_type_key
        patterns_list_to_check = self.synthesis_error_patterns
        if "correctness" in attempt_type_key: # Covers correctness_parameter_reasoning, etc.
            patterns_list_to_check = self.correctness_error_patterns
        
        if not patterns_list_to_check or len(patterns_list_to_check) < self.diversity_trigger_threshold:
            return None # Not enough history or threshold not met for this specific error in its context
            
        # Count occurrences of current_error_signature in the relevant list
        # Only trigger if the *current_error_signature* itself is repeating in its context
        consecutive_count_for_this_error = 0
        for i in range(len(patterns_list_to_check) - 1, -1, -1):
            if patterns_list_to_check[i] == current_error_signature:
                consecutive_count_for_this_error += 1
            else:
                break # Only count consecutive occurrences from the end

        if consecutive_count_for_this_error >= self.diversity_trigger_threshold:
            diversity_key_from_signature = "compile_error" # Default
            
            # Map error signatures to diversity trigger keys
            if "kernelbench_grid_dimensionality" in current_error_signature:
                diversity_key_from_signature = "kernelbench_grid_dimensionality"
            elif "kernelbench_pointer_arithmetic" in current_error_signature:
                diversity_key_from_signature = "kernelbench_pointer_arithmetic"
            elif "kernelbench_masking" in current_error_signature:
                diversity_key_from_signature = "kernelbench_masking"
            elif "kernelbench_runtime" in current_error_signature:
                diversity_key_from_signature = "kernelbench_runtime"
            else:
                # Check for other patterns
                for key_trigger in self.diversity_triggers.keys():
                    if key_trigger in current_error_signature:
                        diversity_key_from_signature = key_trigger
                        break
            
            selected_hint = random.choice(self.diversity_triggers[diversity_key_from_signature])
            
            self.logger.info(
                " DIVERSITY INJECTION (%s context): Error '%s' seen %d times consecutively (threshold: %d). Injecting hint: '%s'",
                attempt_type_key, current_error_signature, consecutive_count_for_this_error, self.diversity_trigger_threshold, selected_hint
            )
            return f"Break the current pattern! {selected_hint}. Previous attempts failed with '{current_error_signature}'. Try a fundamentally different approach for this stage."
        return None

    def _generate_synthesis_error_hint(self, compile_log: Optional[str], research_context: Optional[str]) -> str:
        hint = "The previous kernel source code failed to compile."
        if compile_log:
            hint += f" Compilation log snippet: {compile_log[:500]}" # Truncate for brevity
        if research_context:
            hint += f"\nDEEP RESEARCH CONTEXT: {research_context[:1000]}" # Truncate
        hint += "\nPlease try a different approach to generating the Triton kernel."
        return hint

    def _generate_correctness_feedback(self, corr_out: CorrectOut, last_reasoner_explanation: Optional[str] = None) -> str:
        """Generate detailed feedback for kernel correctness issues with enhanced error analysis."""
        
        if corr_out.correct:
            return "Kernel passed correctness validation."
        
        error_details = corr_out.error_details or {}
        error_type = error_details.get("error_type", "unknown_error")
        error_message = error_details.get("error_message", "No error message provided")
        
        # Base feedback structure
        base_feedback = "Kernel failed correctness validation."
        
        # Enhanced error-specific feedback with actionable solutions
        if error_type == "test_execution_error":
            # Analyze specific test execution errors
            if "tuple index out of range" in error_message:
                feedback = f"{base_feedback} SHAPE INDEXING ERROR: The kernel contains unsafe tensor shape indexing. "
                feedback += "REQUIRED FIXES: 1) Add shape length validation before indexing (e.g., if len(tensor.shape) < 2: raise ValueError), "
                feedback += "2) Use safe indexing patterns like tensor.shape[-1] only after verifying tensor has enough dimensions, "
                feedback += "3) For multi-dimensional inputs, handle different shapes explicitly (2D vs 3D), "
                feedback += "4) In launcher functions, always validate input shapes before processing. "
                feedback += "SPECIFIC SOLUTION: Replace 'M, N = input_tensor.shape[-2], input_tensor.shape[-1]' with safe shape handling."
                
            elif "missing a required argument" in error_message:
                missing_arg = error_message.split("'")[1] if "'" in error_message else "unknown"
                feedback = f"{base_feedback} PARAMETER MISMATCH ERROR: Kernel execution failed because required argument '{missing_arg}' is missing. "
                feedback += "REQUIRED FIXES: 1) Check kernel function signature matches launcher call, "
                feedback += "2) Ensure all kernel parameters are provided in correct order, "
                feedback += "3) Verify constexpr parameters are passed in meta dictionary, "
                feedback += "4) Check that tensor strides are calculated correctly. "
                feedback += f"SPECIFIC SOLUTION: Add missing '{missing_arg}' parameter to kernel call or fix parameter ordering."
                
            elif "All kernel execution attempts failed" in error_message:
                feedback = f"{base_feedback} KERNEL EXECUTION ERROR: Multiple execution strategies failed. "
                feedback += "REQUIRED FIXES: 1) Verify kernel signature matches expected calling patterns, "
                feedback += "2) Check that launcher function handles tensor shapes correctly, "
                feedback += "3) Ensure grid calculation matches kernel expectations, "
                feedback += "4) Validate all pointer arithmetic and memory access patterns. "
                feedback += "DEBUGGING APPROACH: Review kernel parameters, grid launch, and memory layout."
                
            elif "No suitable launcher function found" in error_message:
                feedback = f"{base_feedback} LAUNCHER FUNCTION ERROR: No appropriate launcher function found in kernel code. "
                feedback += "REQUIRED FIXES: 1) Add a launcher function named 'launch_<operation>' (e.g., launch_layer_norm), "
                feedback += "2) Launcher should handle tensor shape validation and grid calculation, "
                feedback += "3) Launcher should call the @triton.jit kernel with correct parameters, "
                feedback += "4) Follow the pattern: def launch_operation(output, input, ...): ... kernel[grid](...). "
                feedback += "TEMPLATE: Provide both @triton.jit kernel and launch_ function."
                
            else:
                feedback = f"{base_feedback} EXECUTION ERROR: {error_message}. "
                feedback += "GENERAL FIXES: 1) Check kernel syntax and imports, "
                feedback += "2) Verify tensor operations are valid, "
                feedback += "3) Ensure proper error handling in launcher function, "
                feedback += "4) Add input validation and shape checks."
        
        elif error_type == "shape_mismatch":
            expected_shape = error_details.get("expected_shape", "unknown")
            actual_shape = error_details.get("actual_shape", "unknown")
            feedback = f"{base_feedback} SHAPE MISMATCH ERROR: Expected output shape {expected_shape}, got {actual_shape}. "
            feedback += "REQUIRED FIXES: 1) Verify grid calculation matches expected output dimensions, "
            feedback += "2) Check tensor reshaping logic in launcher function, "
            feedback += "3) Ensure kernel processes correct number of elements, "
            feedback += "4) Validate output tensor allocation matches expected shape. "
            feedback += f"SPECIFIC SOLUTION: Adjust grid or output tensor creation to produce shape {expected_shape}."
        
        elif error_type == "numerical_mismatch":
            max_diff = error_details.get("max_difference", "unknown")
            tolerance = error_details.get("tolerance_atol", "unknown")
            feedback = f"{base_feedback} NUMERICAL ACCURACY ERROR: Maximum difference {max_diff} exceeds tolerance {tolerance}. "
            feedback += "REQUIRED FIXES: 1) Use higher precision for intermediate calculations, "
            feedback += "2) Implement numerically stable algorithms (e.g., Kahan summation), "
            feedback += "3) Check for proper handling of edge cases (small values, zeros), "
            feedback += "4) Ensure reduction operations maintain precision. "
            feedback += "OPTIMIZATION: Consider using float64 for accumulators or implementing compensated summation."
        
        elif error_type == "invalid_triton_result":
            feedback = f"{base_feedback} INVALID RESULT ERROR: Triton kernel produced NaN or Inf values. "
            feedback += "REQUIRED FIXES: 1) Add checks for division by zero, "
            feedback += "2) Validate input ranges before mathematical operations, "
            feedback += "3) Use epsilon values for numerical stability, "
            feedback += "4) Check for proper initialization of all variables. "
            feedback += "DEBUGGING: Add bounds checking and NaN detection in kernel."
        
        elif error_type == "invalid_pytorch_result":
            feedback = f"{base_feedback} REFERENCE ERROR: PyTorch reference produced invalid results. "
            feedback += "This suggests an issue with the test setup or input generation. "
            feedback += "REQUIRED FIXES: 1) Verify input tensor generation is valid, "
            feedback += "2) Check PyTorch operation parameters, "
            feedback += "3) Ensure test inputs are within valid ranges."
        
        elif error_type == "insufficient_inputs":
            feedback = f"{base_feedback} INPUT ERROR: {error_message}. "
            feedback += "REQUIRED FIXES: 1) Verify operation requires correct number of inputs, "
            feedback += "2) Check input specification matches operation requirements, "
            feedback += "3) Ensure all required tensors are provided."
        
        elif error_type == "unsupported_operation":
            feedback = f"{base_feedback} OPERATION ERROR: {error_message}. "
            feedback += "REQUIRED FIXES: 1) Implement the requested operation type, "
            feedback += "2) Add operation to supported test cases, "
            feedback += "3) Ensure kernel matches expected operation semantics."
        
        elif error_type == "triton_execution_error":
            # Handle specific Triton execution errors
            if "tl.load" in error_message and "offsets" in error_message:
                feedback = f"{base_feedback} MEMORY ACCESS ERROR: Invalid memory access pattern in tl.load. "
                feedback += "REQUIRED FIXES: 1) Verify pointer calculations before tl.load, "
                feedback += "2) Use proper 2D indexing: input_ptr + offs_m[:, None] * stride_m + offs_k[None, :] * stride_k, "
                feedback += "3) Apply correct masking to prevent invalid memory access, "
                feedback += "4) Check tensor strides and memory layout assumptions."
                
            elif "grid" in error_message.lower() and "dimension" in error_message.lower():
                feedback = f"{base_feedback} GRID DIMENSIONALITY ERROR: Kernel grid configuration mismatch. "
                feedback += "REQUIRED FIXES: 1) Match kernel tl.program_id calls with grid dimensions, "
                feedback += "2) Use 1D grid if kernel only uses tl.program_id(0), "
                feedback += "3) Use 2D grid if kernel uses both tl.program_id(0) and tl.program_id(1), "
                feedback += "4) Verify grid calculation matches tensor dimensions."
                
            else:
                feedback = f"{base_feedback} TRITON ERROR: {error_message}. "
                feedback += "GENERAL FIXES: 1) Check Triton language syntax, "
                feedback += "2) Verify kernel parameters and types, "
                feedback += "3) Ensure proper use of Triton operations, "
                feedback += "4) Check memory access patterns and bounds."
        
        elif error_type == "kernelbench_runtime_error":
            specific_runtime_error = error_details.get("specific_runtime_error", error_message)
            feedback = f"{base_feedback} KERNELBENCH RUNTIME ERROR: {specific_runtime_error}. "
            feedback += "REQUIRED FIXES: 1) Check grid launch configuration matches kernel expectations, "
            feedback += "2) Verify all pointer arithmetic includes proper strides, "
            feedback += "3) Ensure masking logic prevents out-of-bounds access, "
            feedback += "4) Use proper 2D indexing patterns for multi-dimensional data."
            
            if specific_runtime_error:
                feedback += f" Specific error: {specific_runtime_error[:200]}..."
        
        elif error_type == "compilation_error":
            feedback = f"Kernel compilation failed. {error_message}. "
            feedback += "REQUIRED FIXES: 1) Fix syntax errors and import statements, "
            feedback += "2) Ensure Triton language compliance, "
            feedback += "3) Check decorator usage (@triton.jit), "
            feedback += "4) Verify constexpr parameter declarations."
        
        else:
            feedback = f"{base_feedback} Error type: {error_type}. {error_message}"
        
        # Add contextual information if available
        if error_details.get("full_traceback"):
            # Extract key lines from traceback for debugging
            traceback_lines = error_details["full_traceback"].split('\n')
            key_lines = [line for line in traceback_lines if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback'])]
            if key_lines:
                feedback += f" TRACEBACK CONTEXT: {'; '.join(key_lines[-3:])}"
        
        # Include reasoner explanation if available
        if last_reasoner_explanation:
            feedback += f" PARAMETER REASONING: {last_reasoner_explanation}"
        
        # Add specific guidance based on error patterns
        feedback += self._add_error_pattern_guidance(error_type, error_message)
        
        return feedback

    def _add_error_pattern_guidance(self, error_type: str, error_message: str) -> str:
        """Add specific guidance based on common error patterns."""
        guidance = ""
        
        # Pattern-based guidance
        if "layer_norm" in error_message.lower():
            guidance += " LAYER_NORM SPECIFIC: Ensure proper handling of normalized_shape parameter and epsilon value. "
            guidance += "Use stable variance calculation: var = mean(x^2) - mean(x)^2 + eps."
        
        elif "matmul" in error_message.lower():
            guidance += " MATMUL SPECIFIC: Verify matrix dimensions are compatible for multiplication. "
            guidance += "Use proper tiling and ensure thread block dimensions match matrix blocking."
        
        elif "softmax" in error_message.lower():
            guidance += " SOFTMAX SPECIFIC: Implement numerically stable softmax with max subtraction. "
            guidance += "Use pattern: x_max = max(x); exp_x = exp(x - x_max); softmax = exp_x / sum(exp_x)."
        
        # Common shape-related issues
        if "shape" in error_message.lower():
            guidance += " SHAPE HANDLING: Always validate tensor dimensions before indexing. "
            guidance += "Use defensive programming: check len(tensor.shape) before accessing specific dimensions."
        
        return guidance

    def _print_correctness_error_details(self, error_details: Optional[dict], synthesis_attempt: int, correctness_attempt: int):
        if not error_details:
            self.logger.debug(f"S{synthesis_attempt}/C{correctness_attempt}: Correctness failed but no error_details provided.")
            return

        self.logger.warning(f"S{synthesis_attempt}/C{correctness_attempt}: Correctness Error Type: {error_details.get('error_type', 'N/A')}")
        
        # Log specific fields if they exist
        if "error_message" in error_details:
            self.logger.warning(f"  Message: {error_details['error_message']}")
        if "mismatched_index" in error_details:
            self.logger.warning(f"  Mismatched Index: {error_details['mismatched_index']}")
        if "expected_value" in error_details and "actual_value" in error_details:
            self.logger.warning(f"  Expected: {error_details['expected_value']}, Actual: {error_details['actual_value']}")
        if "expected_shape" in error_details and "actual_shape" in error_details:
            self.logger.warning(f"  Expected Shape: {error_details['expected_shape']}, Actual Shape: {error_details['actual_shape']}")
        
        # Generic fallback for other details
        other_details = {k: v for k, v in error_details.items() if k not in ['error_type', 'error_message', 'mismatched_index', 'expected_value', 'actual_value', 'expected_shape', 'actual_shape']}
        if other_details:
            self.logger.warning(f"  Other details: {other_details}")

    def _log_pipeline_summary(self, total_synthesis_attempts: int):
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("Total Synthesis Attempts Made: %d / %d", total_synthesis_attempts, self.max_synthesis_attempts)
        # Could add more summary data here, e.g., number of compile errors, correctness errors, etc.
        # For example, count unique error signatures seen:
        if self.synthesis_error_patterns:
            from collections import Counter
            synthesis_counts = Counter(self.synthesis_error_patterns)
            self.logger.info("Synthesis Error Signature Counts: %s", dict(synthesis_counts))
        if self.correctness_error_patterns:
            from collections import Counter
            correctness_counts = Counter(self.correctness_error_patterns)
            self.logger.info("Correctness Error Signature Counts (across all kernels): %s", dict(correctness_counts))
        self.logger.info("=" * 60)

    def _get_error_signature_hash(self, error_signature: str, error_details: Optional[dict] = None) -> str:
        """Generate a consistent hash for an error signature, optionally including details."""
        payload_to_hash = error_signature
        if error_details:
            # Sort dict to ensure consistent hash for same details
            # Convert all values to string to handle potential non-serializable items if any
            try:
                # Only include a few key fields that are likely to be primitive and differentiate errors
                # This avoids hashing large data or objects that might be in error_details
                key_detail_fields = ["error_type", "error_message", "expected_shape", "actual_shape"]
                filtered_details = {k: str(v) for k,v in error_details.items() if k in key_detail_fields}
                if filtered_details:
                    payload_to_hash += json.dumps(filtered_details, sort_keys=True)
            except TypeError as e:
                self.logger.warning(f"Could not serialize error_details for hashing: {e}. Using signature only.")
        
        return hashlib.md5(payload_to_hash.encode()).hexdigest()

    def _should_trigger_deep_research(self, error_signature: str, consecutive_error_count: int, 
                                     is_llm_fallback_mode: bool = False) -> bool:
        if not self.enable_deep_research:
            return False
        if is_llm_fallback_mode:
            return consecutive_error_count >= 1
        return consecutive_error_count >= self.CONSECUTIVE_ERRORS_BEFORE_RESEARCH

    def _get_cached_research_context(self, error_signature: str, error_details: Optional[dict] = None) -> Optional[str]:
        error_key = self._get_error_signature_hash(error_signature, error_details)
        cached_context = self.research_context_cache.get(error_key)
        # session_id and op_hash are not directly available here. Need to pass if required for observer.
        # For now, observer call is simplified.
        obs_metadata = {"error_signature_hash": error_key}
        if cached_context:
            self.logger.info(f"Using cached research context for error signature hash: {error_key}")
            obs_metadata["context_length"] = len(cached_context)
            self.observer.record_observation({"attempt_type": "research_context_cache", "status": "hit", "metadata": obs_metadata})
            return cached_context
        
        self.observer.record_observation({"attempt_type": "research_context_cache", "status": "miss", "metadata": obs_metadata})
        return None

    def _cache_research_context(self, error_signature: str, error_details: Optional[dict], research_context: str):
        error_key = self._get_error_signature_hash(error_signature, error_details)
        self.research_context_cache[error_key] = research_context
        self.logger.info(f"Cached research context for error signature hash: {error_key} (length: {len(research_context)})")
        self.observer.record_observation({"attempt_type": "research_context_cache", "status": "added", 
                                          "metadata": {"error_signature_hash": error_key, "context_length": len(research_context)}})

    @staticmethod
    def _string_to_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        return mapping.get(dtype_str.lower())

    def _profile_pytorch_hotspots(
        self,
        module_src: str,
        init_inputs: Optional[List[Any]],
        input_specs: List[Dict[str, Any]],
        *,
        iterations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run a short torch.profiler session and return top ops by CUDA self time."""
        try:
            # Convert input specs to actual tensors
            input_tensors = []
            for spec in input_specs:
                shape = spec.get("shape", [])
                dtype_str = spec.get("dtype", "float32")
                torch_dtype = self._string_to_torch_dtype(dtype_str)
                if torch_dtype is None:
                    torch_dtype = torch.float32
                
                if not shape:  # Scalar
                    tensor = torch.tensor(1.0, dtype=torch_dtype)
                else:
                    tensor = torch.randn(*shape, dtype=torch_dtype)
                
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                input_tensors.append(tensor)

            # Dynamically build and instantiate the model
            mod = types.ModuleType(f"prof_mod_{id(module_src)}")
            exec(textwrap.dedent(module_src), mod.__dict__)
            if "Model" not in mod.__dict__:
                raise AttributeError("module_src must define a class named 'Model'")
            ModelCls = mod.__dict__["Model"]
            
            # Initialize model with provided init_inputs
            if init_inputs:
                model = ModelCls(*init_inputs)
            else:
                model = ModelCls()
            
            if torch.cuda.is_available():
                model = model.cuda().eval()
            else:
                model = model.eval()

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=iterations, repeat=1),
                record_shapes=True,  # Enhanced: Record tensor shapes for memory analysis
                profile_memory=True,  # Enhanced: Track memory usage
                with_stack=False,
                with_flops=True,  # Enhanced: Track FLOPS when available
            ) as prof:
                with torch.no_grad():
                    for _ in range(iterations + 1):  # +1 for warm-up step
                        model(*input_tensors)
                        prof.step()

            # Summarise
            key_averages = prof.key_averages()
            
            # Check if CUDA timing is available
            has_cuda_timing = len(key_averages) > 0 and hasattr(key_averages[0], 'self_cuda_time_total')
            
            if has_cuda_timing:
                total_time = sum(e.self_cuda_time_total for e in key_averages)
                metric = "self_cuda_time_total"
            else:
                total_time = sum(e.self_cpu_time_total for e in key_averages)
                metric = "self_cpu_time_total"

            def _metric(evt):
                return getattr(evt, metric)

            hotspots: List[Dict[str, Any]] = []
            for evt in sorted(key_averages, key=_metric, reverse=True)[:15]:  # Enhanced: Get top 15 instead of 10
                t_us = _metric(evt)
                pct = (t_us / total_time * 100) if total_time else 0.0
                
                # Enhanced: Create comprehensive hotspot information
                hotspot_info = {
                    "op": evt.key,
                    metric: t_us,
                    "percent": round(pct, 2),
                    "count": evt.count,  # Number of times this operation was called
                    "avg_time": round(t_us / evt.count, 2) if evt.count > 0 else 0.0,
                }
                
                # Enhanced: Add memory information if available
                if hasattr(evt, 'cpu_memory_usage') and evt.cpu_memory_usage > 0:
                    hotspot_info["cpu_memory_usage"] = evt.cpu_memory_usage
                if hasattr(evt, 'cuda_memory_usage') and evt.cuda_memory_usage > 0:
                    hotspot_info["cuda_memory_usage"] = evt.cuda_memory_usage
                
                # Enhanced: Add FLOPS information if available
                if hasattr(evt, 'flops') and evt.flops > 0:
                    hotspot_info["flops"] = evt.flops
                    hotspot_info["gflops_per_sec"] = round((evt.flops / (t_us / 1e6)) / 1e9, 2) if t_us > 0 else 0.0
                
                # Enhanced: Categorize operation type
                op_name_lower = evt.key.lower()
                if any(keyword in op_name_lower for keyword in ["mm", "matmul", "conv", "gemm", "dot", "addmm", "bmm"]):
                    hotspot_info["operation_category"] = "compute_intensive"
                elif any(keyword in op_name_lower for keyword in ["copy", "transpose", "permute", "view", "reshape", "contiguous"]):
                    hotspot_info["operation_category"] = "memory_intensive"
                elif any(keyword in op_name_lower for keyword in ["relu", "gelu", "sigmoid", "tanh", "softmax", "layernorm"]):
                    hotspot_info["operation_category"] = "elementwise"
                else:
                    hotspot_info["operation_category"] = "other"
                
                hotspots.append(hotspot_info)
            
            # Enhanced: Add profiling metadata
            profiling_metadata = {
                "total_profiled_time_us": total_time,
                "total_operations": len(key_averages),
                "iterations": iterations,
                "has_cuda_timing": has_cuda_timing,
                "input_tensor_count": len(input_tensors),
                "input_tensor_shapes": [list(t.shape) for t in input_tensors],
                "model_parameters": sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0,
            }
            
            # Add metadata as the first "hotspot" for easy access
            hotspots.insert(0, {
                "op": "__PROFILING_METADATA__",
                "metadata": profiling_metadata
            })
            
            self.logger.info(f"Enhanced profiling captured {len(hotspots)-1} hotspots from PyTorch execution")
            if len(hotspots) > 1:  # Exclude metadata entry
                top_hotspot = hotspots[1]  # First real hotspot after metadata
                self.logger.info(f"Top hotspot: {top_hotspot['op']} ({top_hotspot['percent']}%, category: {top_hotspot.get('operation_category', 'unknown')})")
            
            return hotspots
            
        except Exception as e:
            self.logger.error(f"Failed to profile PyTorch hotspots: {e}", exc_info=True)
            return []

    async def run(self, op_spec: dict):
        session_id = uuid.uuid4().hex
        op_hash = op_spec.get("op_hash", "unknown")
        self.logger.info("[DIAGNOSTIC LOG] Pipeline run method started for op_hash %s, session_id %s", op_hash, session_id)
        self.logger.info("Pipeline started for op hash %s, session_id %s", op_hash, session_id)

        self.last_performance_feedback = None

        base_observation_data = {
            "session_id": session_id,
            "op_hash": op_hash,
            "input_specs": op_spec.get("input_specs")
        }

        # Gather performance hints for synthesis (once before the loop)
        device_info_for_synthesis: Optional[Dict[str, Any]] = None
        memory_analysis_for_synthesis: Dict[str, Any] = {}
        profiling_hotspots_for_synthesis: Optional[List[Dict[str, Any]]] = None
        try:
            if torch.cuda.is_available():
                proto_analyzer = ProtonRooflineAnalyzer(device="cuda") 
                device_info_for_synthesis = proto_analyzer.get_device_info()
                memory_optimizer = MemoryAccessOptimizer(proto_analyzer.device_specs)
                
                input_specs_list = op_spec.get("input_specs", []) # These are dicts
                if isinstance(input_specs_list, list):
                    for i, spec_dict in enumerate(input_specs_list):
                        shape = spec_dict.get("shape")
                        dtype_str = spec_dict.get("dtype")
                        if shape and dtype_str:
                            torch_dtype = self._string_to_torch_dtype(dtype_str)
                            if torch_dtype:
                                try:
                                    current_shape_tuple = tuple(map(int, shape))
                                    analysis = memory_optimizer.analyze_access_pattern(current_shape_tuple, torch_dtype)
                                    memory_analysis_for_synthesis[f"input_{i}"] = analysis
                                except ValueError as ve:
                                    self.logger.warning(f"Could not parse shape {shape} for input {i} for memory analysis: {ve}")
                                except Exception as e_ma:
                                    self.logger.warning(f"Error analyzing memory access for input {i} ({dtype_str} {shape}): {e_ma}")
                            else:
                                self.logger.warning(f"Could not convert dtype '{dtype_str}' for input {i} for memory analysis.")
                        else:
                            self.logger.warning(f"Input spec {i} missing shape or dtype for memory analysis.")
            else:
                self.logger.warning("CUDA not available, cannot generate device-specific synthesis hints.")
        except RuntimeError as e_cuda:
            self.logger.warning(f"CUDA runtime error during performance hint gathering: {e_cuda}")
        except Exception as e_perf_hints:
            self.logger.error(f"Failed to gather performance hints for synthesis: {e_perf_hints}", exc_info=True)

        # Capture profiling hotspots if module source is available
        try:
            module_src = op_spec.get("module_src")
            init_inputs = op_spec.get("init_inputs")
            if module_src and isinstance(module_src, str):
                self.logger.info("Capturing PyTorch profiling hotspots for synthesis context")
                profiling_hotspots_for_synthesis = self._profile_pytorch_hotspots(
                    module_src, 
                    init_inputs, 
                    op_spec.get("input_specs", [])
                )
                if profiling_hotspots_for_synthesis:
                    self.logger.info(f"Profiling captured {len(profiling_hotspots_for_synthesis)} hotspots - top operation: {profiling_hotspots_for_synthesis[0]['op']} ({profiling_hotspots_for_synthesis[0]['percent']}%)")
                else:
                    self.logger.info("No profiling hotspots captured")
            else:
                self.logger.info("Module source not available - skipping profiling hotspots capture")
        except Exception as e_profiling:
            self.logger.warning(f"Failed to capture profiling hotspots: {e_profiling}")
            profiling_hotspots_for_synthesis = None

        orch_in = {**op_spec, "cache_hit": False}
        orch_out = await self.orchestrator.plan_kernel(OrchestratorIn(**orch_in))
        self.logger.debug("Orchestrator decision | use_cache=%s op_hash=%s", orch_out.use_cache, orch_out.op_hash)

        if orch_out.use_cache:
            mem_out = await self.memory.memory_get(MemoryQueryIn(mode="get", op_hash=orch_out.op_hash))
            self.logger.info("Cache lookup for %s | hit=%s", orch_out.op_hash, mem_out.found)
            if mem_out.found:
                self.observer.record_observation({
                    **base_observation_data,
                    "attempt_type": "cache_lookup",
                    "status": "cache_hit",
                    "kernel_code": mem_out.kernel,
                    "speedup": mem_out.speedup,
                    "latency_ms": mem_out.latency_ms if hasattr(mem_out, 'latency_ms') else None,
                    "metadata": {"source": "cache"}
                })
                result_dict = {
                    "status": "cache_hit",
                    "op_hash": orch_out.op_hash,
                    "kernel": mem_out.kernel,
                    "speedup": mem_out.speedup,
                }
                self.logger.info("Cache hit served | op_hash=%s", orch_out.op_hash)
                return result_dict

        # Initialize synth_in with necessary fields; performance hints are pre-gathered
        # The input_specs from op_spec (list of dicts) are passed here and Pydantic handles conversion.
        synth_in = SynthIn(
            pytorch_src=op_spec["pytorch_src"], 
            input_specs=op_spec["input_specs"], 
            error_hint=None, 
            research_context=None,
            device_info=device_info_for_synthesis, 
            memory_access_analysis=memory_analysis_for_synthesis,
            problem_id=op_spec.get("problem_id"),  # Added for Triton docs knowledge base
            profiling_hotspots=profiling_hotspots_for_synthesis  # Added for PyTorch profiling context
        )
        current_kernel_src: Optional[str] = None
        best_result = {
            "correct": False, 
            "speedup": 0.0, 
            "latency_ms": float('inf'), 
            "kernel": None, 
            "ptx_path": None,
            "source_file_path": None,
            "perf_yaml_path": None,
            "error_details": None,
            "estimated_flops": op_spec.get("estimated_flops"), 
            "estimated_bytes": op_spec.get("estimated_bytes")  
        }
        research_context_str: Optional[str] = None # This can be set by compile or correctness research

        for synthesis_attempt in range(self.max_synthesis_attempts):
            self.logger.info("=" * 60)
            self.logger.info("SYNTHESIS ATTEMPT %d/%d", synthesis_attempt + 1, self.max_synthesis_attempts)
            self.logger.info("=" * 60)
            
            current_error_hint_for_synthesis = synth_in.error_hint
            # research_context_str is from a previous iteration (compile/correctness error research)
            # current_research_context_for_synthesis will use this, or None if it's the first try or was consumed.
            current_research_context_for_synthesis = research_context_str 
            previous_kernel_src_for_synthesis = synth_in.previous_kernel_src

            if self.last_performance_feedback:
                self.logger.info(f"Injecting performance feedback into current synthesis attempt: {self.last_performance_feedback}")
                if current_error_hint_for_synthesis:
                    current_error_hint_for_synthesis = f"{current_error_hint_for_synthesis}. PERFORMANCE FEEDBACK: {self.last_performance_feedback}"
                else:
                    current_error_hint_for_synthesis = f"PERFORMANCE FEEDBACK: {self.last_performance_feedback}"
                if not previous_kernel_src_for_synthesis and best_result["kernel"] and best_result["correct"]:
                     previous_kernel_src_for_synthesis = best_result["kernel"]
                self.last_performance_feedback = None 

            final_synth_in_for_attempt = SynthIn(
                pytorch_src=op_spec["pytorch_src"],
                input_specs=op_spec["input_specs"], 
                expected_output_shape=op_spec.get("expected_output_shape"),
                error_hint=current_error_hint_for_synthesis,
                previous_kernel_src=previous_kernel_src_for_synthesis,
                correctness_hint=synth_in.correctness_hint, 
                research_context=current_research_context_for_synthesis,
                device_info=device_info_for_synthesis,  # Always pass pre-gathered info
                memory_access_analysis=memory_analysis_for_synthesis, # Always pass pre-gathered info
                problem_id=op_spec.get("problem_id"),  # Added for Triton docs knowledge base
                profiling_hotspots=profiling_hotspots_for_synthesis  # Always pass pre-gathered profiling info
            )
            # **CRITICAL FIX**: Only reset hints AFTER synthesis is complete and feedback has been processed
            # Don't reset here as we need to preserve feedback for potential re-attempts
            # The hints will be properly managed at the end of the synthesis loop
            
            # **DEBUG LOGGING**: Track what feedback is being passed to synthesis
            if final_synth_in_for_attempt.correctness_hint:
                self.logger.info(f"🔄 PASSING CORRECTNESS FEEDBACK to synthesis attempt {synthesis_attempt + 1}: {final_synth_in_for_attempt.correctness_hint[:200]}...")
            if final_synth_in_for_attempt.previous_kernel_src:
                self.logger.info(f"🔄 PASSING PREVIOUS KERNEL (length={len(final_synth_in_for_attempt.previous_kernel_src)}) to synthesis attempt {synthesis_attempt + 1}")
            if final_synth_in_for_attempt.error_hint:
                self.logger.info(f"🔄 PASSING ERROR HINT to synthesis attempt {synthesis_attempt + 1}: {final_synth_in_for_attempt.error_hint[:200]}...")
            if final_synth_in_for_attempt.profiling_hotspots:
                self.logger.info(f"🔄 PASSING PROFILING HOTSPOTS to synthesis attempt {synthesis_attempt + 1}: {len(final_synth_in_for_attempt.profiling_hotspots)} operations")
            if not any([final_synth_in_for_attempt.correctness_hint, final_synth_in_for_attempt.previous_kernel_src, final_synth_in_for_attempt.error_hint]):
                self.logger.info(f"🆕 FRESH SYNTHESIS ATTEMPT {synthesis_attempt + 1} (no previous feedback)")

            self.observer.record_observation({
                **base_observation_data,
                "attempt_type": "synthesis_generation",
                "synthesis_attempt_number": synthesis_attempt + 1,
                "status": "initiated",
                "metadata": {
                    "error_hint": final_synth_in_for_attempt.error_hint,
                    "correctness_hint": final_synth_in_for_attempt.correctness_hint,
                    "research_context_present": bool(final_synth_in_for_attempt.research_context),
                    "previous_kernel_src_present": bool(final_synth_in_for_attempt.previous_kernel_src),
                    "device_info_present": bool(final_synth_in_for_attempt.device_info),
                    "memory_analysis_present": bool(final_synth_in_for_attempt.memory_access_analysis),
                    "profiling_hotspots_present": bool(final_synth_in_for_attempt.profiling_hotspots),
                    "profiling_hotspots_count": len(final_synth_in_for_attempt.profiling_hotspots) if final_synth_in_for_attempt.profiling_hotspots else 0
                }
            })

            synth_out = await self.synthesis_agent.synthesize(final_synth_in_for_attempt)
            current_kernel_src = synth_out.kernel_src
            self.logger.debug("Synthesis produced %d chars of kernel code", len(current_kernel_src))
            
            # Update best_result with FLOPS/bytes estimates from synthesis
            if synth_out.estimated_flops is not None:
                best_result["estimated_flops"] = synth_out.estimated_flops
                self.logger.info("Synthesis estimated FLOPS: %.0f", synth_out.estimated_flops)
            if synth_out.estimated_bytes is not None:
                best_result["estimated_bytes"] = synth_out.estimated_bytes
                self.logger.info("Synthesis estimated bytes: %.0f", synth_out.estimated_bytes)
            
            self.observer.record_observation({
                **base_observation_data,
                "attempt_type": "synthesis_generation",
                "synthesis_attempt_number": synthesis_attempt + 1,
                "status": "produced",
                "kernel_code": current_kernel_src,
            })

            compile_in_payload = CompileIn(kernel_src=current_kernel_src)
            comp_out = await self.compile_agent.compile(compile_in_payload)
            self.logger.info("Compile result | ok=%s", comp_out.ok)

            if not comp_out.ok:
                compile_error_signature_for_termination = self._extract_error_signature({
                    "error_type": "compile_error", 
                    "error_message": comp_out.log[:200] if comp_out.log else ""
                })
                self.synthesis_error_patterns.append(compile_error_signature_for_termination)
                
                self.observer.record_observation({
                    **base_observation_data,
                    "attempt_type": "compilation",
                    "synthesis_attempt_number": synthesis_attempt + 1,
                    "status": "compile_error",
                    "kernel_code": current_kernel_src,
                    "compilation_log": comp_out.log,
                    "error_type": "compile_error",
                    "error_message": comp_out.log,
                    "metadata": {"signature_for_termination": compile_error_signature_for_termination}
                })
                
                self._log_attempt_summary(synthesis_attempt + 1, self.max_synthesis_attempts, 
                                        "synthesis", False, compile_error_signature_for_termination)
                
                # Deep research logic for compile errors
                if not research_context_str: # Only do new research if no context from prior correctness failure
                    consecutive_compile_error_count = 0
                    for i in range(len(self.synthesis_error_patterns) - 1, -1, -1):
                        if self.synthesis_error_patterns[i] == compile_error_signature_for_termination:
                            consecutive_compile_error_count += 1
                        else:
                            break
                    
                    compile_error_details_for_research = {
                        "error_type": "compile_error",
                        "error_message": comp_out.log,
                        # Potentially add src_snippet if available from comp_out or synth_out
                        "src_snippet": current_kernel_src[:500] # Example: pass a snippet
                    }

                    research_context_str = self._get_cached_research_context(
                        compile_error_signature_for_termination,
                        compile_error_details_for_research
                    )

                    if not research_context_str and self._should_trigger_deep_research(
                        compile_error_signature_for_termination,
                        consecutive_compile_error_count
                        # is_llm_fallback_mode could be True if compile errors often lead to fallback
                    ):
                        self.logger.warning(
                            f"Compile error pattern '{compile_error_signature_for_termination}' repeated {consecutive_compile_error_count} times. "
                            f"Threshold ({self.CONSECUTIVE_ERRORS_BEFORE_RESEARCH}) met. Invoking Deep Research Agent for compile error."
                        )
                        research_query_compile = (
                            f"The following Triton kernel failed to compile. Error details: {comp_out.log}. "
                            f"The error signature is '{compile_error_signature_for_termination}'. "
                            "Please provide a detailed explanation of this compilation error, common causes, "
                            "and specific, actionable solutions with corrected Triton code examples. "
                            "Focus on resolving the syntax or API usage issue in the kernel."
                        )
                        try:
                            if hasattr(self.deep_research_manager, 'fetch_research_summary'):
                                research_context_str = await self.deep_research_manager.fetch_research_summary(
                                    query=research_query_compile,
                                    error_details=compile_error_details_for_research,
                                    web_search_func=self.web_search_tool_func
                                )
                                if research_context_str:
                                    self._cache_research_context(
                                        compile_error_signature_for_termination,
                                        compile_error_details_for_research,
                                        research_context_str
                                    )
                                    self.logger.info("Deep research provided context for compile error (first 300 chars): %s...", research_context_str[:300])
                                    self.observer.record_observation({
                                        **base_observation_data,
                                        "attempt_type": "deep_research_compile_error",
                                        "synthesis_attempt_number": synthesis_attempt + 1,
                                        "status": "context_obtained",
                                        "metadata": {"query": research_query_compile, "context_length": len(research_context_str)}
                                    })
                                else:
                                    self.logger.warning("Deep research for compile error ran but returned no context.")
                            else:
                                self.logger.error("DeepResearchManager missing 'fetch_research_summary' – research for compile error skipped.")
                        except Exception as e_dr_compile:
                            self.logger.error("Deep research for compile error invocation failed: %s", e_dr_compile, exc_info=True)
                
                # Check for termination *before* invoking reasoner if research didn't change the plan
                # If research provided new context, we might want to try reasoner and one more synth
                if self._should_terminate_early(compile_error_signature_for_termination, "synthesis") and not research_context_str : # if research did nothing, and we should terminate.
                    self.logger.warning(f"Early termination condition met for synthesis error: {compile_error_signature_for_termination}. Skipping reasoner and further synthesis for this path.")
                    self.observer.record_observation({
                        **base_observation_data,
                        "attempt_type": "pipeline_control",
                        "synthesis_attempt_number": synthesis_attempt + 1,
                        "status": "terminated_early_compile_error_no_research_effect",
                        "error_type": compile_error_signature_for_termination
                    })
                    break # Out of the synthesis loop

                self.logger.warning("Compilation failed. Invoking reasoner for fix. Research context available: %s", "Yes" if research_context_str else "No")
                
                reasoner_in_payload = ReasonerIn(
                        compile_log=comp_out.log or "", 
                        kernel_src_to_analyze=comp_out.full_kernel_src if comp_out.full_kernel_src else current_kernel_src,
                        research_context=research_context_str # Pass research context if available
                    )
                reason = await self.compile_reasoner_agent.reason(reasoner_in_payload)
                final_hint = reason.fix_hint
                
                # Prepare SynthIn for the next attempt (retry current synthesis_attempt number due to compile fail)
                # This is setting up `synth_in` which is used at the START of the next loop iteration.
                synth_in.error_hint = final_hint
                synth_in.previous_kernel_src = current_kernel_src
                synth_in.correctness_hint = None # Clear correctness hint from any prior successful compile
                # research_context_str here would be from a correctness failure's deep research, if any.
                # If compilation failed, research_context_str would have been handled by the compile failure block.
                # If research was done for this compile error and given to reasoner, it's consumed for this hint.
                # The global `research_context_str` will be cleared before next *full* attempt, or if new research occurs.
                synth_in.research_context = research_context_str # This ensures it's available for the next immediate synthesis if we continue
                # Performance hints are static for this op_spec run, always pass them.
                synth_in.device_info = device_info_for_synthesis
                synth_in.memory_access_analysis = memory_analysis_for_synthesis
                synth_in.profiling_hotspots = profiling_hotspots_for_synthesis  # Always pass profiling info
                
                # If early termination was triggered *after* research/reasoning attempt (e.g. still same error)
                if self._should_terminate_early(compile_error_signature_for_termination, "synthesis"):
                    self.logger.warning(f"Early termination condition met for synthesis error: {compile_error_signature_for_termination} post-reasoning. Terminating synthesis attempts.")
                    self.observer.record_observation({
                        **base_observation_data,
                        "attempt_type": "pipeline_control",
                        "synthesis_attempt_number": synthesis_attempt + 1,
                        "status": "terminated_early_compile_error_post_reasoning",
                        "error_type": compile_error_signature_for_termination
                    })
                    break # Out of the synthesis loop

                research_context_str = None # Clear research context before the next full synthesis attempt
                                        # It would be repopulated if deep research is triggered for compile/correctness in that attempt.
                continue # To the next iteration of the FOR loop, retrying the current synthesis_attempt effectively

            self.observer.record_observation({
                **base_observation_data,
                "attempt_type": "compilation",
                "synthesis_attempt_number": synthesis_attempt + 1,
                "status": "success",
                "kernel_code": current_kernel_src,
                "kernel_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                "compilation_log": comp_out.log
            })

            kernel_source_for_this_compilation = current_kernel_src
            if comp_out.ptx_path:
                reasoning_key_path = str(comp_out.ptx_path)
            else:
                reasoning_key_path = f"kernel_src_hash_{hash(kernel_source_for_this_compilation)}"
                self.logger.warning(f"ptx_path was None for a compiled kernel. Using hash as key: {reasoning_key_path}")

            self.logger.info(f"Initializing reasoning history for key: {reasoning_key_path}")
            await self.correctness_reasoner_memory.process(
                CorrectnessReasonerMemoryIn(
                    operation="clear_history",
                    clear_payload=CorrectnessReasonerMemoryClearIn(
                        key=CorrectnessReasonerMemoryKey(kernel_source_path=reasoning_key_path)
                    )
                )
            )

            suggested_params_from_reasoner = None
            last_correctness_result = None
            last_reasoner_explanation = None

            for reasoning_attempt_num in range(self.max_parameter_reasoning_attempts):
                self.logger.info("-" * 40)
                self.logger.info(
                    "PARAMETER REASONING ATTEMPT %d/%d (Synthesis %d)",
                    reasoning_attempt_num + 1,
                    self.max_parameter_reasoning_attempts,
                    synthesis_attempt + 1
                )
                self.logger.info("Kernel key: %s", reasoning_key_path)
                self.logger.info("-" * 40)

                corr_in_payload = CorrectIn(
                    ptx_path=comp_out.ptx_path,
                    pytorch_src=op_spec["pytorch_src"],
                    input_specs=op_spec["input_specs"],
                    op_params=op_spec.get("op_params"),
                    suggested_execution_params=suggested_params_from_reasoner,
                    problem_id=op_spec.get("problem_id"),
                    level=op_spec.get("level")
                )
                corr_out = await self.correctness_agent.validate(corr_in_payload)
                last_correctness_result = corr_out

                current_error_signature = None
                obs_status = "unknown_correctness_status"
                obs_error_type = None
                obs_error_message = None
                obs_error_details = None

                if not corr_out.correct:
                    obs_status = "correctness_error"
                    if corr_out.error_details:
                        current_error_signature = self._extract_error_signature(corr_out.error_details)
                        self.correctness_error_patterns.append(current_error_signature)
                        obs_error_type = current_error_signature
                        obs_error_message = corr_out.error_details.get("error_message")
                        obs_error_details = corr_out.error_details
                        
                        if obs_error_type and "missing_arg" in obs_error_type:
                             obs_status = "runtime_error_missing_arg"
                        elif obs_error_type == "shape_mismatch":
                             obs_status = "correctness_error_shape_mismatch"
                        elif obs_error_type == "value_mismatch":
                             obs_status = "correctness_error_value_mismatch"
                        elif obs_error_type == "invalid_grid_syntax":
                             obs_status = "runtime_error_invalid_grid"
                        elif obs_error_type:
                             obs_status = f"runtime_error_{obs_error_type}"

                    self._log_attempt_summary(reasoning_attempt_num + 1, self.max_parameter_reasoning_attempts,
                                            "correctness", False, current_error_signature)
                    self._print_correctness_error_details(corr_out.error_details, synthesis_attempt + 1, reasoning_attempt_num + 1)
                    
                    # Invoke Deep Research Agent if repeated correctness errors occur before early termination/parameter reasoning
                    if not research_context_str and current_error_signature:
                        # Count how many times this exact correctness error has appeared consecutively
                        consecutive_corr_error_count = 0
                        for i in range(len(self.correctness_error_patterns) - 1, -1, -1):
                            if self.correctness_error_patterns[i] == current_error_signature:
                                consecutive_corr_error_count += 1
                            else:
                                break
                        
                        # Check cache first
                        research_context_str = self._get_cached_research_context(
                            current_error_signature, 
                            corr_out.error_details
                        )
                        
                        # Trigger deep-research when the configurable threshold is reached and research will be used
                        if not research_context_str and self._should_trigger_deep_research(
                            current_error_signature, 
                            consecutive_corr_error_count,
                            is_llm_fallback_mode=True  # Correctness errors often lead to LLM fallback
                        ):
                            self.logger.warning(
                                f"Correctness error pattern '{current_error_signature}' repeated {consecutive_corr_error_count} times. "
                                f"Threshold ({self.CONSECUTIVE_ERRORS_BEFORE_RESEARCH}) met. Invoking Deep Research Agent."
                            )
                            research_query = (
                                f"The following Triton kernel produced incorrect runtime results. Error details: {corr_out.error_details}. "
                                f"The error signature is '{current_error_signature}'. "
                                "Please provide a detailed explanation of this error, common causes, "
                                "and specific, actionable solutions with corrected Triton code examples. "
                                "Focus on resolving the underlying numerical or logic issue in the kernel."
                            )
                            try:
                                if hasattr(self.deep_research_manager, 'fetch_research_summary'):
                                    research_context_str = await self.deep_research_manager.fetch_research_summary(
                                        query=research_query, 
                                        error_details=corr_out.error_details, # Pass structured corr.error_details
                                        web_search_func=self.web_search_tool_func
                                    )
                                    if research_context_str:
                                        # Cache the research context
                                        self._cache_research_context(
                                            current_error_signature,
                                            corr_out.error_details,
                                            research_context_str
                                        )
                                        
                                        self.logger.info("Deep research provided context (first 300 chars): %s...", research_context_str[:300])
                                        
                                        # Record observation for deep research
                                        self.observer.record_observation({
                                            **base_observation_data,
                                            "attempt_type": "deep_research",
                                            "synthesis_attempt_number": synthesis_attempt + 1,
                                            "sub_attempt_number": reasoning_attempt_num + 1,
                                            "status": "context_obtained",
                                            "metadata": {"query": research_query, "context_length": len(research_context_str)}
                                        })
                                    else:
                                        self.logger.warning("Deep research ran but returned no context.")
                                else:
                                    self.logger.error("DeepResearchManager missing 'fetch_research_summary' – research skipped.")
                            except Exception as e_dr:
                                self.logger.error("Deep research invocation failed: %s", e_dr, exc_info=True)
                    
                    if current_error_signature and self._should_terminate_early(current_error_signature, "correctness"):
                        self.logger.error("Early termination due to repeated correctness errors")
                        self.observer.record_observation({
                            **base_observation_data,
                            "attempt_type": "pipeline_control",
                            "synthesis_attempt_number": synthesis_attempt + 1,
                            "sub_attempt_number": reasoning_attempt_num + 1,
                            "status": "terminated_early",
                            "error_type": "repeated_correctness_error",
                            "metadata": {"triggering_signature": current_error_signature, "kernel_key": reasoning_key_path}
                        })
                        break
                else:
                    obs_status = "success"
                    self._log_attempt_summary(reasoning_attempt_num + 1, self.max_parameter_reasoning_attempts,
                                            "correctness", True)

                self.observer.record_observation({
                    **base_observation_data,
                    "attempt_type": "correctness_validation",
                    "synthesis_attempt_number": synthesis_attempt + 1,
                    "correctness_attempt_number": reasoning_attempt_num + 1,
                    "status": obs_status,
                    "kernel_code": kernel_source_for_this_compilation,
                    "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                    "source_file_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                    "latency_ms": corr_out.latency_ms if corr_out.correct else None,
                    "speedup": corr_out.speedup if corr_out.correct else None,
                    "error_type": obs_error_type,
                    "error_message": obs_error_message,
                    "error_details": obs_error_details,
                    "error_signature": current_error_signature if not corr_out.correct else None,
                    "suggested_params_used": suggested_params_from_reasoner,
                    "metadata": {"kernel_key": reasoning_key_path}
                })

                await self.memory.memory_put(
                    MemoryPutIn(
                        mode="put",
                        op_hash=orch_out.op_hash,
                        kernel=kernel_source_for_this_compilation,
                        latency_ms=corr_out.latency_ms,
                        speedup=corr_out.speedup,
                        ptx_path=str(comp_out.ptx_path) if comp_out.ptx_path else None,
                    )
                )

                if best_result["correct"] is False or \
                   (corr_out.correct and (not best_result["correct"] or corr_out.speedup > best_result["speedup"])):
                    best_result = {
                        "correct": corr_out.correct,
                        "speedup": corr_out.speedup,
                        "kernel": kernel_source_for_this_compilation,
                        "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                        "latency_ms": corr_out.latency_ms,
                        "synthesis_attempt": synthesis_attempt + 1,
                        "reasoning_attempt": reasoning_attempt_num + 1,
                        "source_file_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                        "perf_yaml_path": None,
                        "error_details": corr_out.error_details,
                        "estimated_flops": op_spec.get("estimated_flops"),
                        "estimated_bytes": op_spec.get("estimated_bytes")
                    }
                    self.logger.info(f"New best result tracked: correct={corr_out.correct}, speedup={corr_out.speedup:.2f}")
                    self.observer.record_observation({
                        **base_observation_data,
                        "attempt_type": "best_result_update",
                        "synthesis_attempt_number": synthesis_attempt + 1,
                        "sub_attempt_number": reasoning_attempt_num + 1,
                        "status": "new_best_found",
                        "kernel_code": kernel_source_for_this_compilation,
                        "ptx_path": best_result["ptx_path"],
                        "latency_ms": best_result["latency_ms"],
                        "speedup": best_result["speedup"],
                        "metadata": {"is_correct": best_result["correct"]}
                    })

                if corr_out.correct:
                    self.logger.info("Kernel is CORRECT. Proceeding to Performance Analysis.")
                    
                    if not comp_out.source_file_path or not Path(comp_out.source_file_path).exists():
                        self.logger.error(f"Performance Analysis SKIPPED: Source file path from CompileOut is invalid or file does not exist: {comp_out.source_file_path}")
                        if not best_result["correct"] or (corr_out.latency_ms < best_result["latency_ms"] and best_result["latency_ms"] != float('inf')):
                             best_result.update({
                                "correct": True, "speedup": corr_out.speedup, "latency_ms": corr_out.latency_ms,
                                "kernel": kernel_source_for_this_compilation, "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                                "source_file_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                                "perf_yaml_path": None,
                                "error_details": {"performance_analysis_skipped": "invalid_source_file_path"}
                            })
                             self.logger.info(f"New best (but perf-skipped due to invalid source file) result updated using correctness agent's metrics: speedup={corr_out.speedup:.2f}x, latency={corr_out.latency_ms:.3f}ms")
                        self.last_performance_feedback = "Performance analysis skipped: invalid source file for profiling. Cannot provide detailed hints for synthesis."
                        break # Kernel is correct, but cannot be profiled. Break correctness loop.

                    perf_in = PerformanceIn(
                        op_hash=op_hash,
                        ptx_path=str(comp_out.ptx_path) if comp_out.ptx_path else "", # Ensure string
                        source_file_path=str(comp_out.source_file_path), # Already checked
                        input_specs=op_spec["input_specs"],
                        pytorch_src=op_spec["pytorch_src"],
                        runs=op_spec.get("perf_runs", 20),
                        estimated_flops=best_result.get("estimated_flops"),
                        estimated_bytes_transferred=best_result.get("estimated_bytes")
                    )
                    perf_out = await self.performance_agent.profile(perf_in)
                    
                    self.logger.info(f"Performance Analysis completed | Triton Runtime: {perf_out.runtime_ms:.3f}ms, Speedup vs Eager: {perf_out.speedup:.2f}x, Efficiency: {perf_out.efficiency_percent:.1f}%")
                    if perf_out.yaml_path:
                        self.logger.info(f"Performance YAML report: {perf_out.yaml_path}")
                    else:
                        self.logger.warning("Performance YAML path was not generated or returned by PerformanceAgent.")

                    self.observer.record_observation({
                        **base_observation_data,
                        "attempt_type": "performance_analysis",
                        "synthesis_attempt_number": synthesis_attempt + 1,
                        "correctness_attempt_number": reasoning_attempt_num + 1, 
                        "status": "completed" if perf_out.runtime_ms > 0 else "failed_or_no_data",
                        "kernel_code": kernel_source_for_this_compilation, 
                        "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                        "source_file_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                        "perf_runtime_ms": perf_out.runtime_ms, "perf_speedup": perf_out.speedup,
                        "perf_efficiency_percent": perf_out.efficiency_percent, 
                        "perf_occupancy_percent": perf_out.occupancy_percent,
                        "perf_yaml_path": perf_out.yaml_path, 
                        "perf_recommendations": perf_out.recommendations,
                        "metadata": {"device_specs_from_perf": perf_out.device_specs, 
                                     "kernel_metrics_raw_from_perf": perf_out.kernel_metrics_raw}
                    })

                    if perf_out.runtime_ms > 0: # Successful profiling
                        if not best_result["correct"] or perf_out.speedup > best_result["speedup"]:
                            best_result.update({
                                "correct": True, "speedup": perf_out.speedup, "latency_ms": perf_out.runtime_ms,
                                "kernel": kernel_source_for_this_compilation, "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                                "source_file_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                                "perf_yaml_path": perf_out.yaml_path, "error_details": None
                            })
                            self.logger.info(f"New best kernel based on performance analysis: speedup={perf_out.speedup:.2f}x, latency={perf_out.runtime_ms:.3f}ms, efficiency={perf_out.efficiency_percent:.1f}%")
                        
                        if perf_out.speedup < self.target_speedup_threshold or perf_out.efficiency_percent < self.min_efficiency_for_no_hint:
                            rec_str = " ".join(perf_out.recommendations[:2]) if perf_out.recommendations else "No specific recommendations provided by profiler."
                            self.last_performance_feedback = f"Prior kernel was correct. Achieved Speedup {perf_out.speedup:.2f}x (target >{self.target_speedup_threshold:.2f}x), Efficiency {perf_out.efficiency_percent:.1f}% (target >{self.min_efficiency_for_no_hint:.1f}%). Key Suggestions: {rec_str}"
                            self.logger.info(f"Performance suboptimal. Storing feedback for next synthesis attempt: {self.last_performance_feedback}")
                        else:
                            self.logger.info(f"Performance meets targets (Speedup: {perf_out.speedup:.2f}x, Efficiency: {perf_out.efficiency_percent:.1f}%). Cleared any prior performance feedback.")
                            self.last_performance_feedback = None 
                    else: # Performance profiling itself failed
                        self.logger.warning("Performance profiling failed or returned invalid/error data. Using CorrectnessAgent's metrics for this attempt if no prior correct kernel exists.")
                        if not best_result["correct"]:
                             best_result.update({
                                "correct": True, "speedup": corr_out.speedup,
                                "latency_ms": corr_out.latency_ms,
                                "kernel": kernel_source_for_this_compilation, "ptx_path": str(comp_out.ptx_path) if comp_out.ptx_path else None,
                                "source_file_path": str(comp_out.source_file_path) if comp_out.source_file_path else None,
                                "perf_yaml_path": perf_out.yaml_path, # Path to error YAML from perf agent
                                "error_details": {"performance_profiling_failed": True, 
                                                  "details": perf_out.recommendations[0] if perf_out.recommendations else "Unknown profiling error"}
                            })
                             self.logger.info(f"New best result (kernel correct, but performance profiling failed): using correctness metrics - speedup={corr_out.speedup:.2f}x, latency={corr_out.latency_ms:.3f}ms")
                        self.last_performance_feedback = f"Performance profiling failed: {perf_out.recommendations[0] if perf_out.recommendations else 'Internal profiler error'}. Cannot provide detailed optimization hints for synthesis based on this profiling attempt."

                    # Update memory cache if this kernel is the current best_result
                    if best_result["correct"] and best_result["kernel"] == kernel_source_for_this_compilation:
                        mem_payload = MemoryPutIn(
                            mode="put", op_hash=op_hash, kernel=best_result["kernel"],
                            latency_ms=best_result["latency_ms"], 
                            speedup=best_result["speedup"], 
                            ptx_path=str(best_result["ptx_path"]) # Ensure ptx_path is string
                        )
                        await self.memory.memory_put(mem_payload)
                        self.logger.info(f"Correct kernel (speedup {best_result['speedup']:.2f}x) with performance data stored/updated in memory cache.")
                    
                    break # Break from correctness attempts loop (kernel correct and profiled/handled)

                else: # Correctness check failed (corr_out.correct is False)
                    current_error_signature = self._extract_error_signature(corr_out.error_details or {})
                    self.correctness_error_patterns.append(current_error_signature)
                    self._log_attempt_summary(reasoning_attempt_num + 1, self.max_parameter_reasoning_attempts,
                                              "correctness", False, current_error_signature)
                    self._print_correctness_error_details(corr_out.error_details, synthesis_attempt + 1, reasoning_attempt_num + 1)
                    
                    correctness_hint = self._generate_correctness_feedback(corr_out, last_reasoner_explanation)
                    
                    # **CRITICAL FIX**: Set the correctness hint for the next synthesis attempt
                    synth_in.correctness_hint = correctness_hint
                    synth_in.previous_kernel_src = kernel_source_for_this_compilation
                    
                    # Log that feedback will be passed to next synthesis
                    self.logger.info(f"Correctness feedback prepared for next synthesis: {correctness_hint[:200]}...")
                    
                    # Inject diversity hint if error pattern is repeating
                    if current_error_signature:
                        diversity_hint = self._inject_diversity_hint(
                            current_error_signature, 
                            "correctness_parameter_reasoning", 
                            reasoning_attempt_num + 1
                        )
                        if diversity_hint:
                            # Combine correctness feedback with diversity hint
                            synth_in.correctness_hint = f"{correctness_hint} DIVERSITY INJECTION: {diversity_hint}"
                            self.logger.info(f"Added diversity hint to correctness feedback for synthesis")
                    
                    # Record failed attempt for CorrectnessReasonerMemory
                    if reasoning_key_path: # Ensure key is valid for memory
                        # The suggested_params_from_reasoner might be from a *previous* reasoner call if this is not the first correctness attempt.
                        # For the first correctness attempt, it will be None.
                        grid_tried = suggested_params_from_reasoner.get("grid_config", "default_or_unknown") if suggested_params_from_reasoner else "initial_default"
                        args_tried = suggested_params_from_reasoner.get("kernel_args", []) if suggested_params_from_reasoner else ["initial_default"]
                        
                        failed_attempt_detail = ReasoningAttemptDetail(
                            suggested_grid=grid_tried,
                            suggested_args=args_tried,
                            error_received=str(corr_out.error_details) if corr_out.error_details else "Unknown correctness error"
                        )
                        await self.correctness_reasoner_memory.process(
                            CorrectnessReasonerMemoryIn(
                                operation="add_attempt", 
                                add_payload=CorrectnessReasonerMemoryAddIn(
                                    key=CorrectnessReasonerMemoryKey(kernel_source_path=reasoning_key_path),
                                    attempt=failed_attempt_detail
                                )
                            )
                        )

                    if self._should_terminate_early(current_error_signature, "correctness_parameter_reasoning"):
                        self.logger.warning(f"EARLY TERMINATION (Correctness Loop): Max correctness error patterns for '{current_error_signature}' met. Moving to next synthesis attempt.")
                        synth_in.error_hint = f"Previous kernel version failed correctness repeatedly with signature '{current_error_signature}'. Last error: {correctness_hint}"
                        synth_in.previous_kernel_src = kernel_source_for_this_compilation
                        break # Break from correctness loop, go to next synthesis attempt
                
                # If max correctness attempts reached for this kernel version and still not correct
                if reasoning_attempt_num == self.max_parameter_reasoning_attempts - 1 and not corr_out.correct:
                    self.logger.warning(f"Max parameter reasoning attempts ({self.max_parameter_reasoning_attempts}) reached for kernel. It remains incorrect.")
                    synth_in.error_hint = f"Previous kernel ({kernel_source_for_this_compilation[:60]}...) failed all {self.max_parameter_reasoning_attempts} correctness/reasoning attempts. Last error: {correctness_hint}"
                    synth_in.previous_kernel_src = kernel_source_for_this_compilation
                    # **ENSURE**: Correctness hint is also set here
                    synth_in.correctness_hint = correctness_hint
            
            # --- After Correctness Loop for the current synthesized kernel ---
            if best_result["correct"] and best_result["kernel"] == kernel_source_for_this_compilation and not self.last_performance_feedback:
                 self.logger.info(f"SUCCESS: Correct and performant kernel found for op_hash {op_hash}. Speedup: {best_result['speedup']:.2f}x. Terminating synthesis loop.")
                 self._log_pipeline_summary(synthesis_attempt + 1)
                 
                 # Run final benchmark comparison
                 benchmark_plot_path = await self._run_final_benchmark(best_result, op_spec, session_id)
                 if benchmark_plot_path:
                     best_result["benchmark_plot_path"] = benchmark_plot_path
                 
                 return await self._prepare_final_result(op_hash, "success_max_attempts_best_effort", synthesis_attempt + 1, best_result, fallback_if_needed=False, session_id=session_id)

            # Prepare for the NEXT synthesis attempt (incrementing synthesis_attempt number)
            # `synth_in` at this stage holds state from the completed attempt (e.g., if it failed correctness)
            current_synth_error_hint = synth_in.error_hint 
            current_synth_prev_kernel = synth_in.previous_kernel_src
            # research_context_str here would be from a correctness failure's deep research, if any.
            # If compilation failed, research_context_str would have been handled by the compile failure block.
            current_synth_research_context = research_context_str

            if self.last_performance_feedback: 
                self.logger.info(f"Preparing for next synthesis attempt with performance feedback: {self.last_performance_feedback}")
                perf_hint = f"PERFORMANCE REVISION NEEDED: {self.last_performance_feedback}"
                current_synth_error_hint = f"{current_synth_error_hint}. {perf_hint}" if current_synth_error_hint else perf_hint
                current_synth_prev_kernel = current_kernel_src # The kernel that was correct but slow
                current_synth_research_context = None # Perf feedback usually doesn't need old research for compile/correctness errors
            
            # Inject diversity hint for synthesis-level errors if patterns are repeating
            if len(self.synthesis_error_patterns) > 0:
                last_synthesis_error_sig = self.synthesis_error_patterns[-1]
                diversity_hint = self._inject_diversity_hint(
                    last_synthesis_error_sig, 
                    "synthesis_agent_level", 
                    synthesis_attempt + 1
                )
                if diversity_hint:
                    self.logger.info(f"Injecting synthesis-level diversity hint for next attempt: {diversity_hint}")
                    current_synth_error_hint = f"{current_synth_error_hint}. DIVERSITY HINT: {diversity_hint}" if current_synth_error_hint else f"DIVERSITY HINT: {diversity_hint}"
            
            # Update the main synth_in object for the next full synthesis attempt
            synth_in.error_hint = current_synth_error_hint
            synth_in.previous_kernel_src = current_synth_prev_kernel
            synth_in.research_context = current_synth_research_context 
            # Performance hints are static for this op_spec run, they were set before the loop.
            synth_in.device_info = device_info_for_synthesis
            synth_in.memory_access_analysis = memory_analysis_for_synthesis
            
            # **CRITICAL FIX**: Reset hints appropriately based on iteration state
            # If we're moving to the next synthesis attempt due to success, clear correctness hint
            # If we're moving due to failure, preserve the correctness hint for the next synthesis
            if best_result["correct"] and best_result["kernel"] == kernel_source_for_this_compilation:
                # Success case - clear all hints for next attempt
                synth_in.correctness_hint = None
                synth_in.error_hint = None
                synth_in.previous_kernel_src = None
            # Otherwise, preserve the correctness_hint that was set during correctness failures
            
            research_context_str = None # Clear research context before the next full synthesis attempt
                                        # It would be repopulated if deep research is triggered for compile/correctness in that attempt.

        # --- End of Synthesis Loop (Max attempts reached) ---
        self.logger.warning(f"Max synthesis attempts ({self.max_synthesis_attempts}) reached for op_hash {op_hash}.")
        self._log_pipeline_summary(self.max_synthesis_attempts)
        
        # Run final benchmark if we have a correct kernel
        if best_result["correct"] and best_result["kernel"]:
            self.logger.info("Running final benchmark for best correct kernel found")
            benchmark_plot_path = await self._run_final_benchmark(best_result, op_spec, session_id)
            if benchmark_plot_path:
                best_result["benchmark_plot_path"] = benchmark_plot_path
        
        final_status_msg = "success_max_attempts_best_effort" if best_result["correct"] else "failure_max_attempts_no_correct_kernel"
        return await self._prepare_final_result(op_hash, final_status_msg, self.max_synthesis_attempts, best_result, 
                                          fallback_if_needed=not best_result["correct"], session_id=session_id)

    async def _prepare_final_result(self, op_hash: str, status_message: str, attempts_taken: int, 
                                   best_result_data: dict, fallback_if_needed: bool = False, session_id: str = "unknown_session") -> dict:
        """Helper to prepare the final dictionary to be returned by the pipeline."""
        
        # Debug logging to help diagnose issues
        self.logger.debug(f"_prepare_final_result called with best_result_data type: {type(best_result_data)}, value: {best_result_data}")
        
        current_datetime_iso = datetime.datetime.now().isoformat() # For consistent timestamping

        if fallback_if_needed and not (isinstance(best_result_data, dict) and best_result_data.get("correct")): 
            self.logger.error(f"No correct kernel found for {op_hash} after {attempts_taken} attempts. Invoking fallback agent.")
            fb_in = FallbackIn(correct=False, speedup=0.0, op_hash=op_hash)
            fb_out = await self.fallback_agent.fallback(fb_in) 
            self.logger.info(f"Fallback agent provided kernel with speedup {fb_out.speedup:.2f}x for op_hash {op_hash}")
            
            self.observer.record_observation({
                "session_id": session_id, "op_hash": op_hash,
                "attempt_type": "fallback_agent_invoked", "synthesis_attempt_number": attempts_taken, 
                "status": "completed", "kernel_code": fb_out.final_kernel, "speedup": fb_out.speedup, 
                "metadata": {"reason": "No correct kernel found by pipeline after max attempts", "timestamp": current_datetime_iso}
            })
            return {
                "status": "failure_fallback_provided", "op_hash": op_hash,
                "kernel": fb_out.final_kernel, "speedup": fb_out.speedup, "latency_ms": -1.0, 
                "ptx_path": None, "source_file_path": None, "perf_yaml_path": None, 
                "synthesis_attempts": attempts_taken,
                "error_details": {"log": "Max attempts reached, no correct kernel found. Fallback kernel used.", "timestamp": current_datetime_iso}
            }
        else: 
            final_status = status_message
            best_result_correct = isinstance(best_result_data, dict) and best_result_data.get("correct")
            if not best_result_correct and status_message == "success_max_attempts_best_effort":
                final_status = "failure_max_attempts_no_correct_kernel"
            elif not best_result_correct and status_message == "success":
                final_status = "failure_despite_early_success_status_pipeline_logic_error"
                self.logger.error(f"Pipeline state inconsistency for op_hash {op_hash}: status claims 'success' but best_result is not correct.")

            best_result_speedup = best_result_data.get('speedup', 0.0) if isinstance(best_result_data, dict) else 0.0
            self.logger.info(f"Pipeline finished for {op_hash}. Final Status: {final_status}. Best kernel correct: {best_result_correct}, Speedup: {best_result_speedup:.2f}x")
            
            # Sanitize best_result_data for observation metadata to avoid large/complex objects
            safe_best_result_details = {}
            if isinstance(best_result_data, dict) and best_result_data is not None:
                try:
                    safe_best_result_details = { 
                        k: (str(v)[:200] if isinstance(v, Path) else v) # Truncate paths, keep others
                        for k, v in best_result_data.items() 
                        if not isinstance(v, torch.Tensor) and k not in ["kernel"] # Exclude tensors and full kernel code
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to sanitize best_result_data: {e}. Using empty dict.")
                    safe_best_result_details = {}
            if isinstance(best_result_data, dict) and best_result_data is not None and best_result_data.get("kernel"):
                 safe_best_result_details["kernel_snippet"] = str(best_result_data.get("kernel"))[:200] + "..."

            error_details_for_obs = best_result_data.get("error_details") if isinstance(best_result_data, dict) else None
            if isinstance(error_details_for_obs, dict): # Ensure error_details_for_obs is a dict before trying to iterate
                error_details_for_obs = {k: v for k,v in error_details_for_obs.items() if not isinstance(v, torch.Tensor)}
            elif error_details_for_obs is not None: # If not dict but not None, convert to string
                error_details_for_obs = str(error_details_for_obs)


            self.observer.record_observation({
                "session_id": session_id, "op_hash": op_hash,
                "attempt_type": "pipeline_final_result", 
                "synthesis_attempt_number": attempts_taken,
                "status": final_status, 
                "kernel_code_snippet": safe_best_result_details.get("kernel_snippet"), 
                "speedup": best_result_data.get("speedup") if isinstance(best_result_data, dict) else None, 
                "latency_ms": best_result_data.get("latency_ms") if isinstance(best_result_data, dict) else None,
                "error_details_summary": error_details_for_obs, 
                "metadata": {
                    "final_status_message": final_status, 
                    "best_result_details_summary": safe_best_result_details, 
                    "timestamp": current_datetime_iso
                }
            })

            # Ensure safe_best_result_details is never None to prevent unpacking errors
            if safe_best_result_details is None:
                safe_best_result_details = {}
            
            # Safely extract error_details from safe_best_result_details
            existing_error_details = safe_best_result_details.get("error_details", {}) if isinstance(safe_best_result_details, dict) else {}
            if not isinstance(existing_error_details, dict):
                existing_error_details = {}

            return {
                "status": final_status,
                "op_hash": op_hash,
                "kernel": best_result_data.get("kernel") if isinstance(best_result_data, dict) else None,
                "speedup": best_result_data.get("speedup") if isinstance(best_result_data, dict) else None,
                "latency_ms": best_result_data.get("latency_ms") if isinstance(best_result_data, dict) else None,
                "ptx_path": best_result_data.get("ptx_path") if isinstance(best_result_data, dict) else None,
                "source_file_path": best_result_data.get("source_file_path") if isinstance(best_result_data, dict) else None,
                "perf_yaml_path": best_result_data.get("perf_yaml_path") if isinstance(best_result_data, dict) else None,
                "benchmark_plot_path": best_result_data.get("benchmark_plot_path") if isinstance(best_result_data, dict) else None,
                "synthesis_attempts": attempts_taken,
                "error_details": {**existing_error_details, "final_status_message": final_status, "timestamp": current_datetime_iso}
            }

    async def _run_final_benchmark(self, best_result: dict, op_spec: dict, session_id: str) -> Optional[str]:
        """
        Run final benchmarking comparison between Triton kernel and PyTorch reference.
        Returns the path to the generated benchmark plot, or None if benchmarking fails.
        """
        if not best_result["correct"] or not best_result["kernel"]:
            self.logger.info("Skipping final benchmark - no correct kernel available")
            return None
        
        try:
            import triton.testing
            import matplotlib.pyplot as plt
            import tempfile
            import importlib.util
            from pathlib import Path
            
            if op_spec.get('problem_id') == 23:
                self.logger.info("🏁 Running SOFTMAX final performance benchmark comparison (Problem ID 23)")
            else:
                self.logger.info("🏁 Running final performance benchmark comparison")
            
            # Load the Triton kernel
            kernel_src = best_result["kernel"]
            
            # Create temporary file for the kernel
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    temp_file_path = f.name
                    f.write(kernel_src)
                    f.flush()
                    
                    # Import the kernel module
                    spec = importlib.util.spec_from_file_location("triton_kernel", f.name)
                    triton_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(triton_module)
                
                # --- Force GPU device for the imported Triton kernel module (if available) ---
                if torch.cuda.is_available():
                    try:
                        torch.cuda.set_device(0)
                    except Exception:
                        pass

                    if hasattr(triton_module, 'DEVICE'):
                        try:
                            import torch as _torch_internal
                            triton_module.DEVICE = _torch_internal.device('cuda:0')
                            self.logger.info("Patched Triton module DEVICE to cuda:0 explicitly")
                        except Exception as e:
                            self.logger.warning(f"Unable to patch Triton module DEVICE: {e}")
                # ---------------------------------------------------------------------------
                
                # === Use built-in Triton benchmark for fused softmax when available ===
                if op_spec.get('problem_id') == 23 and hasattr(triton_module, 'benchmark_softmax'):
                    self.logger.info("Using built-in benchmark_softmax from the kernel module to generate the performance graph (mirrors Triton docs).")
                    try:
                        # Run the reference benchmark (saves a file named e.g. 'softmax-performance.png').
                        triton_module.benchmark_softmax.run(show_plots=False, print_data=True, save_path=".")

                        from pathlib import Path
                        built_plots = list(Path(".").glob("softmax-performance*.png"))
                        if built_plots:
                            plot_path = built_plots[0]
                            output_dir = Path("performance_output")
                            output_dir.mkdir(exist_ok=True)
                            final_plot = output_dir / f"{session_id}_triton-vs-torch-performance-problem-23.png"
                            import shutil
                            shutil.move(str(plot_path), final_plot)
                            self.logger.info(f"📊 Benchmark plot saved to: {final_plot}")
                            return str(final_plot)
                        else:
                            self.logger.warning("benchmark_softmax completed but no plot file was found – defaulting to generic benchmarking logic.")
                    except Exception as e:
                        self.logger.error(f"benchmark_softmax failed with error: {e}. Falling back to generic benchmarking.")
                # === End built-in benchmark path ===
                
                # Find the launcher function
                launcher_func = None
                available_functions = [name for name in dir(triton_module) if not name.startswith('_')]
                self.logger.info(f"Available functions in kernel module: {available_functions}")
                
                # Look for specific function names based on operation type
                if op_spec.get('problem_id') == 23:  # Softmax
                    if hasattr(triton_module, 'softmax'):
                        launcher_func = triton_module.softmax
                        self.logger.info(f"Found softmax launcher function")
                    
                # Fallback to launch_ prefix functions
                if not launcher_func:
                    for name in dir(triton_module):
                        if name.startswith('launch_'):
                            launcher_func = getattr(triton_module, name)
                            self.logger.info(f"Found launcher function: {name}")
                            break
                
                if not launcher_func:
                    self.logger.warning("No launcher function found in kernel - skipping benchmark")
                    self.logger.warning(f"Available functions were: {available_functions}")
                    return None
                
                # Determine benchmark parameters based on input specs
                input_specs = op_spec["input_specs"]
                if not input_specs:
                    self.logger.warning("No input specs available - skipping benchmark")
                    return None
                
                primary_input = input_specs[0]
                input_shape = primary_input["shape"]
                
                # Set up benchmark parameters based on operation type and input shape
                if len(input_shape) == 2:  # 2D tensor (like softmax)
                    M, N = input_shape
                    
                    # For softmax (problem ID 23), use parameters matching 02-fused-softmax.py
                    if op_spec.get('problem_id') == 23:
                        # Use fixed M=4096 and vary N like in 02-fused-softmax.py
                        x_vals = [128 * i for i in range(2, 100)]  # Exact range from 02-fused-softmax.py
                        benchmark_args = {'M': 4096}  # Fixed M like in 02-fused-softmax.py
                    else:
                        # Create benchmark with varying N dimension for other operations
                        x_vals = [128 * i for i in range(2, min(50, N // 128 + 10))]  # Reasonable range
                        benchmark_args = {'M': M}
                    
                    x_name = 'N'
                    ylabel = "GB/s"
                    plot_name = f"triton-vs-torch-performance-problem-{op_spec.get('problem_id', 'unknown')}"
                elif len(input_shape) == 1:  # 1D tensor
                    N = input_shape[0]
                    x_vals = [1024 * i for i in range(1, min(20, N // 1024 + 5))]
                    benchmark_args = {}
                    x_name = 'N'
                    ylabel = "GB/s"
                    plot_name = f"triton-vs-torch-performance-problem-{op_spec.get('problem_id', 'unknown')}"
                else:
                    # For higher dimensional tensors, use total elements
                    total_elements = 1
                    for dim in input_shape:
                        total_elements *= dim
                    x_vals = [1024 * i for i in range(1, min(20, total_elements // 1024 + 5))]
                    benchmark_args = {'total_elements_base': total_elements}  # Pass as base value
                    x_name = 'total_elements'
                    ylabel = "GB/s"
                    plot_name = f"triton-vs-torch-performance-problem-{op_spec.get('problem_id', 'unknown')}"
                
                # Create PyTorch reference function
                def create_pytorch_reference():
                    """Create PyTorch reference function from op_spec."""
                    pytorch_src = op_spec["pytorch_src"]
                    
                    # Handle different PyTorch source formats
                    if "torch.softmax" in pytorch_src:
                        # For softmax, use axis=-1 to match 02-fused-softmax.py reference implementation
                        def pytorch_reference(x):
                            return torch.softmax(x, axis=-1)
                        return pytorch_reference
                    else:
                        # Generic approach for other operations
                        ref_func_src = f"""
import torch
import torch.nn.functional as F

def pytorch_reference(x):
    {pytorch_src.replace('return ', '').strip()}
    return result if 'result' in locals() else x
"""
                        
                        local_scope = {}
                        exec(ref_func_src, {"torch": torch, "F": torch.nn.functional}, local_scope)
                        return local_scope['pytorch_reference']
                
                pytorch_ref = create_pytorch_reference()
                
                # Define benchmark function
                @triton.testing.perf_report(
                    triton.testing.Benchmark(
                        x_names=[x_name],
                        x_vals=x_vals,
                        line_arg='provider',
                        line_vals=['triton', 'torch'],
                        line_names=["Triton", "PyTorch"],
                        styles=[('blue', '-'), ('green', '-')],
                        ylabel=ylabel,
                        plot_name=plot_name,
                        args=benchmark_args,
                    ))
                def benchmark_func(**kwargs):
                    provider = kwargs.pop('provider')
                    
                    # Create input tensor based on benchmark parameters
                    # Prefer the DEVICE specified in the imported Triton module (mirrors
                    # official Triton tutorial style). Fallback to first CUDA device or CPU.
                    if hasattr(triton_module, 'DEVICE'):
                        device = triton_module.DEVICE  # torch.device instance from kernel file
                    else:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    # Ensure the chosen CUDA device is activated so subsequent allocations /
                    # launches run on the expected GPU context.
                    if isinstance(device, torch.device) and device.type == 'cuda':
                        try:
                            torch.cuda.set_device(device)
                        except Exception:
                            # Default to cuda:0 if specified device index cannot be set
                            torch.cuda.set_device(0)

                    if len(input_shape) == 2:
                        M = kwargs.get('M', input_shape[0])
                        N = kwargs.get('N', input_shape[1])
                        x = torch.randn(M, N, device=device, dtype=torch.float32)
                    elif len(input_shape) == 1:
                        N = kwargs.get('N', input_shape[0])
                        x = torch.randn(N, device=device, dtype=torch.float32)
                    else:
                        # For higher dimensional tensors, create based on total elements
                        total_elements = kwargs.get('total_elements', kwargs.get('total_elements_base', 1024))
                        x = torch.randn(total_elements, device=device, dtype=torch.float32)
                        # Reshape to original shape if possible
                        try:
                            x = x.view(*input_shape)
                        except:
                            pass  # Keep flat if reshape fails
                    
                    # Set up device stream (matching reference implementation)
                    device_type = x.device.type
                    if hasattr(torch, device_type):
                        stream = getattr(torch, device_type).Stream()
                        getattr(torch, device_type).set_stream(stream)
                    
                    # GB/s calculation function (matching reference implementation)
                    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
                    
                    if provider == 'torch':
                        ms = triton.testing.do_bench(lambda: pytorch_ref(x))
                    elif provider == 'triton':
                        # Check launcher function signature to determine how to call it
                        import inspect
                        sig = inspect.signature(launcher_func)
                        param_names = list(sig.parameters.keys())
                        
                        # Handle specific known launcher patterns
                        if launcher_func.__name__ == 'softmax':
                            # softmax(x) - takes input, returns output (from 02-fused-softmax.py)
                            self.logger.info(f"Benchmarking softmax with input shape {x.shape}")
                            ms = triton.testing.do_bench(lambda: launcher_func(x))
                        elif launcher_func.__name__ == 'launch_softmax':
                            # launch_softmax(x, output=None) - takes input and optional output
                            self.logger.info(f"Benchmarking launch_softmax with input shape {x.shape}")
                            ms = triton.testing.do_bench(lambda: launcher_func(x))
                        elif len(param_names) == 1:
                            # Launcher takes only input, returns output: launch_func(x)
                            ms = triton.testing.do_bench(lambda: launcher_func(x))
                        elif len(param_names) >= 2 and 'output' in param_names[0].lower():
                            # Launcher takes output + inputs: launch_func(output, x)
                            output = torch.empty_like(x)
                            ms = triton.testing.do_bench(lambda: launcher_func(output, x))
                        else:
                            # Default: try single input first, then output + input
                            try:
                                ms = triton.testing.do_bench(lambda: launcher_func(x))
                            except:
                                output = torch.empty_like(x)
                                ms = triton.testing.do_bench(lambda: launcher_func(output, x))
                    else:
                        return 0.0
                    
                    # Return GB/s using the lambda function (matching reference implementation)
                    return gbps(ms)
                
                # Run benchmark and save plot
                self.logger.info(f"Running benchmark with {len(x_vals)} data points...")
                self.logger.info(f"Benchmark parameters: x_name={x_name}, x_vals={x_vals[:5]}..., plot_name={plot_name}")
                benchmark_func.run(show_plots=False, print_data=True, save_path=".")
                
                # Find the generated plot file
                plot_files = list(Path(".").glob(f"{plot_name}*.png"))
                if plot_files:
                    plot_path = str(plot_files[0])
                    self.logger.info(f"✅ Benchmark plot saved to: {plot_path}")
                    
                    # Move to performance_output directory
                    output_dir = Path("performance_output")
                    output_dir.mkdir(exist_ok=True)
                    final_plot_path = output_dir / f"{session_id}_{plot_name}.png"
                    
                    import shutil
                    shutil.move(plot_path, final_plot_path)
                    self.logger.info(f"📊 Final benchmark plot: {final_plot_path}")
                    
                    return str(final_plot_path)
                else:
                    self.logger.warning("Benchmark plot file not found")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Failed to run final benchmark: {e}", exc_info=True)
                return None
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass  # Ignore cleanup errors
                
        except Exception as e:
            self.logger.error(f"Failed to run final benchmark: {e}", exc_info=True)
            # Clean up temporary file in case of exception
            if 'temp_file_path' in locals() and temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass  # Ignore cleanup errors
            return None
