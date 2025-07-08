from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any

# -------------------- Shared --------------------
class TensorSpec(BaseModel):
    shape: List[int]
    dtype: Literal["float16", "float32", "int32", "int8"]

class OpSpec(BaseModel):
    problem_id: int               # KernelBench row id
    level: int                    # Difficulty level 1–4
    pytorch_src: str              # Reference PyTorch forward() implementation
    input_specs: List[TensorSpec]
    op_params: Optional[Dict[str, Any]] = None  # Parameters for model initialization (e.g., eps for RMSNorm)
    # Added for profiling support
    module_src: Optional[str] = None  # Full module source code for profiling
    init_inputs: Optional[List[Any]] = None  # Initialization inputs for model construction

# -------------------- 1. Orchestrator --------------------
class OrchestratorIn(OpSpec):
    cache_hit: bool

class OrchestratorOut(BaseModel):
    op_hash: str
    backend: Literal["triton"]
    use_cache: bool

# -------------------- 2. Memory --------------------
class MemoryQueryIn(BaseModel):
    mode: Literal["get"]
    op_hash: str

class MemoryQueryOut(BaseModel):
    found: bool
    kernel: Optional[str] = None
    speedup: Optional[float] = None

class MemoryPutIn(BaseModel):
    mode: Literal["put"]
    op_hash: str
    kernel: str
    latency_ms: float
    speedup: float
    ptx_path: str

# Schemas for MemoryAgent itself if it were to be called as a single LlmAgent entrypoint
class MemoryAgentIn(BaseModel):
    operation: Literal["get", "put"]
    query_payload: Optional[MemoryQueryIn] = None
    put_payload: Optional[MemoryPutIn] = None

class MemoryAgentOut(BaseModel):
    query_out: Optional[MemoryQueryOut] = None

# -------------------- 3. Synthesis --------------------
class SynthIn(BaseModel):
    pytorch_src: str
    input_specs: List[TensorSpec]
    expected_output_shape: Optional[List[int]] = None
    error_hint: Optional[str] = None
    previous_kernel_src: Optional[str] = None
    correctness_hint: Optional[str] = None
    research_context: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None  # Added for performance-aware synthesis
    memory_access_analysis: Optional[Dict[str, Any]] = None  # Added for performance-aware synthesis
    problem_id: Optional[int] = None  # Added for Triton docs knowledge base integration
    profiling_hotspots: Optional[List[Dict[str, Any]]] = None  # Added for PyTorch profiling context

class SynthOut(BaseModel):
    kernel_src: str
    estimated_flops: Optional[float] = None
    estimated_bytes: Optional[float] = None

# -------------------- 4. Compile --------------------
class CompileIn(BaseModel):
    kernel_src: str

class CompileOutOK(BaseModel):
    ok: Literal[True]
    ptx_path: str

class CompileOutFail(BaseModel):
    ok: bool = False
    log: str
    src_snippet_for_log: str
    full_kernel_src: str

# New CompileOut to unify OK and Fail scenarios for schema definition
class CompileOut(BaseModel):
    ok: bool
    ptx_path: Optional[str] = None
    source_file_path: Optional[str] = None  # Path to the saved kernel source file
    log: Optional[str] = None
    src_snippet_for_log: Optional[str] = None
    full_kernel_src: Optional[str] = None

# -------------------- 5. Reasoner --------------------
class ReasonerIn(BaseModel):
    compile_log: str
    kernel_src_to_analyze: str
    research_context: Optional[str] = None

class ReasonerOut(BaseModel):
    fix_hint: str

# -------------------- 6. Correctness + Timer --------------------
class CorrectIn(BaseModel):
    ptx_path: str
    pytorch_src: str
    input_specs: List[TensorSpec]
    runs: int = 10
    suggested_execution_params: Optional[dict] = None
    op_params: Optional[Dict[str, Any]] = None # Added field for operator parameters
    problem_id: Optional[int] = None # KernelBench problem ID for direct model loading
    level: Optional[int] = None # KernelBench level for direct model loading
    module_src: Optional[str] = None  # Full module source code for better reference model creation

class CorrectOut(BaseModel):
    correct: bool
    latency_ms: float
    speedup: float
    error_details: Optional[dict] = None  # For debugging failed correctness checks

# -------------------- 6.5. Correctness Reasoner --------------------
class ReasoningAttemptDetail(BaseModel):
    suggested_grid: str
    suggested_args: List[str]
    error_received: str

class CorrectnessReasonerIn(BaseModel):
    kernel_source_path: str
    input_specs: List[TensorSpec]
    expected_output_shape: List[int]
    error_message: str
    previous_reasoning_attempts: Optional[List[ReasoningAttemptDetail]] = None

class CorrectnessReasonerOut(BaseModel):
    calling_pattern: str
    grid_config: str
    kernel_args: List[str]
    explanation: str

# -------------------- 6.6. Correctness Reasoner Memory --------------------
class CorrectnessReasonerMemoryKey(BaseModel):
    kernel_source_path: str

class CorrectnessReasonerMemoryAddIn(BaseModel):
    key: CorrectnessReasonerMemoryKey
    attempt: ReasoningAttemptDetail # This reuses the existing ReasoningAttemptDetail

class CorrectnessReasonerMemoryGetIn(BaseModel):
    key: CorrectnessReasonerMemoryKey

class CorrectnessReasonerMemoryClearIn(BaseModel):
    key: CorrectnessReasonerMemoryKey

class CorrectnessReasonerMemoryGetOut(BaseModel):
    history: Optional[List[ReasoningAttemptDetail]] = None
    success: bool
    message: Optional[str] = None

class CorrectnessReasonerMemoryWriteOut(BaseModel):
    success: bool
    message: Optional[str] = None

# Umbrella input model for the agent
class CorrectnessReasonerMemoryIn(BaseModel):
    operation: Literal["add_attempt", "get_history", "clear_history"]
    add_payload: Optional[CorrectnessReasonerMemoryAddIn] = None
    get_payload: Optional[CorrectnessReasonerMemoryGetIn] = None
    clear_payload: Optional[CorrectnessReasonerMemoryClearIn] = None

# Umbrella output model for the agent (can be simplified if distinct outputs are preferred)
class CorrectnessReasonerMemoryOut(BaseModel):
    get_history_result: Optional[CorrectnessReasonerMemoryGetOut] = None
    write_result: Optional[CorrectnessReasonerMemoryWriteOut] = None

# -------------------- 7. Fallback --------------------
class FallbackIn(BaseModel):
    correct: bool
    speedup: float
    op_hash: str

class FallbackOut(BaseModel):
    final_kernel: str
    speedup: float

# Schemas for EvaluatorAgent
class EvaluatorIn(BaseModel):
    level: int
    limit: Optional[int] = None

class EvaluatorOut(BaseModel):
    results_json: str

# -------------------- 8. Performance Analyzer --------------------
class PerformanceIn(BaseModel):
    op_hash: str
    ptx_path: str
    source_file_path: str
    input_specs: List[TensorSpec]
    pytorch_src: str
    runs: int = 10
    estimated_flops: Optional[float] = None
    estimated_bytes_transferred: Optional[float] = None

class PerformanceOut(BaseModel):
    runtime_ms: float
    speedup: float
    achieved_gflops: float
    achieved_bandwidth_gb_s: float
    efficiency_percent: float
    occupancy_percent: float # Added as per design
    recommendations: List[str]
    yaml_path: str
    # Include raw device specs and kernel metrics for detailed YAML
    device_specs: Optional[Dict[str, Any]] = None 
    kernel_metrics_raw: Optional[Dict[str, Any]] = None
    raw_proton_metrics: Optional[Dict[str, float]] = None # Added for detailed proton output
    profile_hatchet_path: Optional[str] = None # Path to the .hatchet file from proton
    memory_access_analysis: Optional[Dict[str, Any]] = None # Analysis from MemoryAccessOptimizer

# -------------------- DB Models (not agent contracts but used by Memory) --------------------
# ... existing code ... 