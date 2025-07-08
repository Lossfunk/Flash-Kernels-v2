# Project Daniel – Triton-Driven Kernel Synthesis Pipeline

**Table of contents**
1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Triton version compatibility](#triton-version-compatibility)
4. [Installation & setup](#installation--setup)
5. [Agents & orchestration flow](#agents--orchestration-flow)
6. [Running KernelBench](#running-kernelbench)
7. [End-to-end example](#end-to-end-example)
8. [Troubleshooting](#troubleshooting)

---

## Overview
This workspace demonstrates a *self-refining GPU-kernel pipeline* built
around **Triton 2/3**.  A set of cooperating **agents** iteratively
synthesize, compile, benchmark and reason about Triton kernels for tasks
originating from [KernelBench](https://github.com/openai/kernelbench).

The whole stack is coded in Python and requires **no C/CUDA
boiler-plate** – Triton and a few LLM endpoints do the heavy lifting.

```mermaid
graph TD
    A([Start]):::start --> B["OrchestratorAgent: Plan"]:::agent
    B --> C{"MemoryAgent: Cache Get"}:::decision
    C -- "Cache Hit":::hit --> D["Return Cached Kernel"]:::cache
    D --> Z_HIT(["End: Cache Hit"]):::success
    C -- "Cache Miss":::miss --> E["Loop: Max N Synthesis Attempts"]:::loop

    E --> F["SynthesisAgent: Generate Kernel<br/>(receives hints: compile, correctness, research, diversity, performance)"]:::agent
    F --> G["CompileAgent: Compile<br/>(Triton → PTX/JIT)"]:::agent
    G --> H{"Compile OK?"}:::decision

    H -- "No":::no --> H_NO_DEC{"Repeated Error?"}:::decision
    H_NO_DEC -- "Yes":::yes --> H_DR["DeepResearchManager: Get Context"]:::agent
    H_DR --> H_CR["CompileReasonerAgent (+Research): Get Fix Hint"]:::agent
    H_NO_DEC -- "No":::no --> H_CR
    H_CR --> F

    H -- "Yes":::yes --> J1["CorrectnessAgent: Validate"]:::agent
    J1 --> J1A["Try KernelBench Validation<br/>(Build PyTorch + ModelNew modules)"]:::validation
    J1A --> J1B{"KernelBench<br/>Success?"}:::decision
    
    J1B -- "Yes":::yes --> J2["Correctness OK?"]:::decision
    J1B -- "No (Compilation Error)":::no --> J1C["Fallback: Direct Triton Validation<br/>(Execute kernel directly vs PyTorch reference)"]:::validation
    J1C --> J2

    J2 -- "No":::no --> J_NO_DEC{"Max Tuning Attempts<br/>or Repeated Error?"}:::decision
    J_NO_DEC -- "No":::no --> J_CRM_GET["CorrectnessReasonerMemoryAgent: Get History"]:::agent
    J_CRM_GET --> J_CR_PARAM["CorrectnessReasonerAgent: Suggest Params"]:::agent
    J_CR_PARAM --> J_CRM_ADD["CorrectnessReasonerMemoryAgent: Add Attempt"]:::agent
    J_CRM_ADD --> J1
    J_NO_DEC -- "Yes (tuning failed for this kernel)":::fail_loop --> F_HINT["Provide Correctness Hint to SynthesisAgent"]:::info
    F_HINT -.-> E

    J2 -- "Yes":::yes --> Perf_A["PerformanceAgent: Profile & Analyze<br/>(Proton profiling, roofline analysis)"]:::agent
    Perf_A --> Perf_CHK{"Profiling<br/>Successful?"}:::decision
    Perf_CHK -- "No (Proton session error)":::no --> Perf_FB_ERR["Use Timing-Only Metrics<br/>+ Provide Error Feedback"]:::info
    Perf_FB_ERR -.-> Perf_TGT
    Perf_CHK -- "Yes":::yes --> Perf_TGT{"Performance Target Met?"}:::decision
    Perf_A -. "log" .-> DB
    Perf_CHK -. "log" .-> DB

    Perf_TGT -- "Yes":::yes --> K["MemoryAgent: Cache Put"]:::cache
    K --> PREP_SUCCESS["Prepare Final Result<br/>(Safe data validation & formatting)"]:::validation
    PREP_SUCCESS --> M["Return Result<br/>(Correct & Performant Kernel<br/>with Perf. Metrics & Report)"]:::result
    M --> Z_LOOP(["End: Attempt Success<br/>(Correct & Performant)"]):::success

    Perf_TGT -- "No":::no --> Perf_FB["Provide Performance Feedback<br/>to SynthesisAgent"]:::info
    Perf_FB -.-> E
    Perf_FB -. "log" .-> DB

    E -- "All Synthesis Attempts Failed or Terminated Early":::fail --> N["FallbackAgent: Final Decision<br/>(use best found or fail)"]:::fallback
    N -- "Best Result Found":::hit --> PREP_BEST["Prepare Final Result<br/>(Safe data handling & validation)"]:::validation
    PREP_BEST --> O_BEST["Return Best Cached Kernel"]:::cache
    O_BEST --> Z_BEST(["End: Best Result Success"]):::success
    N -- "No Correct Kernel":::miss --> PREP_FAIL["Prepare Final Result<br/>(Safe error handling)"]:::validation
    PREP_FAIL --> O["Return Failed/Fallback"]:::failed
    O --> Z_FAIL(["End: All Failed"]):::failure

    DB[("Experience DB<br/>(SQLite via Observer)<br/>Logs: synthesis, compilation, validation,<br/>errors, hints, performance metrics")]:::db_node_style

    F -. "log" .-> DB
    G -. "log" .-> DB
    H_CR -. "log" .-> DB
    H_DR -. "log" .-> DB
    J1 -. "log" .-> DB
    J1A -. "log" .-> DB
    J1C -. "log" .-> DB
    J_CR_PARAM -. "log" .-> DB
    K -. "log" .-> DB
    D -. "log" .-> DB
    PREP_SUCCESS -. "log" .-> DB
    PREP_BEST -. "log" .-> DB
    PREP_FAIL -. "log" .-> DB
    M -. "log" .-> DB
    O -. "log" .-> DB
    O_BEST -. "log" .-> DB

    classDef start fill:#FFD700,stroke:#FF8C00,stroke-width:2px,color:black
    classDef agent fill:#5DADE2,stroke:#2E86C1,stroke-width:2px,color:white
    classDef decision fill:#58D68D,stroke:#27AE60,stroke-width:2px,color:black
    classDef cache fill:#AAB7B8,stroke:#7F8C8D,stroke-width:2px,color:black
    classDef loop fill:#F7DC6F,stroke:#D4AC0D,stroke-width:2px,color:black
    classDef fallback fill:#F39C12,stroke:#B9770E,stroke-width:2px,color:white
    classDef result fill:#7DCEA0,stroke:#27AE60,stroke-width:2px,color:black
    classDef failed fill:#EC7063,stroke:#CB4335,stroke-width:2px,color:white
    classDef success fill:#27AE60,stroke:#196F3D,stroke-width:2px,color:white,stroke-dasharray:5 5
    classDef failure fill:#C0392B,stroke:#7B241C,stroke-width:2px,color:white,stroke-dasharray:5 5
    classDef db_node_style fill:#AED6F1,stroke:#3498DB,stroke-width:2px,color:black
    classDef info fill:#FADBD8,stroke:#AEB6BF,stroke-width:1px,color:black
    classDef validation fill:#E8DAEF,stroke:#8E44AD,stroke-width:2px,color:black
    classDef hit stroke:#27AE60,stroke-width:2px,color:#27AE60
    classDef miss stroke:#F39C12,stroke-width:2px,color:#F39C12
    classDef yes stroke:#2E86C1,stroke-width:2px,color:#2E86C1
    classDef no stroke:#E74C3C,stroke-width:2px,color:#E74C3C
    classDef fail stroke:#C0392B,stroke-width:2px,color:#C0392B
    classDef fail_loop stroke:#C0392B,stroke-width:2px,color:#C0392B,stroke-dasharray:2 2
```

**Note:** All agent interactions, decisions, and outcomes in the flow above are logged to a persistent SQLite "Experience Database" (`data/kernel_pipeline_observations.sqlite`) for analysis and to enable future learning capabilities.

## Repository structure
```
Project_daniel/
├─ agents/                 # Autonomous building blocks (LLM, compile …)
│  ├─ base.py              # Common utilities
│  ├─ compile/agent.py     # Kernel → PTX/JIT compiler
│  ├─ correctness/agent.py # Execute unit/perf checks
│  ├─ reasoner/agent.py    # Produce hints when compilation fails
│  ├─ synthesis/agent.py   # LLM-driven kernel generator
│  ├─ observers.py         # Data logging component (e.g., to SQLite)
│  └─ …
│
├─ utils/                 # Logging, dataclasses, GenAI client, …
├─ data/                  # Directory for persistent data, e.g., SQLite DBs
│  └─ kernel_pipeline_observations.sqlite # Database of all pipeline events
├─ KernelBench/           # Upstream benchmark suite (git-subtree)
├─ rc.py                  # Stand-alone compiler helper (shows fallbacks)
├─ run_kernelbench.py     # CLI: run pipeline over KernelBench levels
├─ tmp_* directories      # Auto-saved payloads, sources, outputs
└─ requirements.txt       # Pinned Python dependencies (Triton ≥2,<3)
```

### Docs & assets
* `docs/overview_flow.svg` – high-level diagram (produced from the PNG
  in the figure above – edit if needed)

## Triton version compatibility
The codebase supports **both Triton 2.3** (current PyTorch wheels).  All compile paths try the modern API first and
automatically fall back to the 2.x ASTSource path when required.  This
is handled in:

* `rc.py` – reference implementation for a single kernel file.
* `agents/compile/agent.py` – production compiler agent.

No manual action is needed – just ensure *exactly one* Triton version is
visible in your environment (`pip list | grep triton`).

## Installation & setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (Optional) set your preferred LLM providers – see utils/genai_client.py
export GOOGLE_API_KEY="…"  
LOG_LEVEL=DEBUG (OR INFO)  # for Gemini
```

> ⚠️  A CUDA-capable GPU with compute capability ≥ 70 is required for the
> default kernels.

### Dataset / benchmark assets
The repository already ships the KernelBench prompts.  No further data
is required.

## Agents & orchestration flow
The orchestration logic lives in `agents/pipeline_orchestrator.py`.  A
single problem specification flows through the following stages:

| Stage | Agent | Responsibility |
|-------|-------|----------------|
| 1 | **SynthesisAgent** | Generate an initial Triton kernel given the PyTorch reference & tensor specs. |
| 2 | **CompileAgent** | Convert Triton source → PTX (and optionally CUDA binary); handles 2.x/3.x API differences, caches inputs/outputs. |
| 3 | **CorrectnessAgent** | Execute the kernel, compare against reference implementation, measure latency. |
| 4 | **ReasonerAgent** | When compilation or tests fail, request an LLM hint (via GenAI client). |
| 5 | **Loop** | The orchestrator feeds the hint back to the synthesizer and repeats until success or max-iters. |

All inter-agent messages are json-serialisable Pydantic models defined in
`agents/contracts.py`.

**Data Logging:** Throughout this process, detailed observations about each attempt (synthesis, compilation, correctness, errors, hints, performance metrics) are recorded by an `SQLiteObserver` into a persistent database (`data/kernel_pipeline_observations.sqlite`). This creates an "Experience Database" which is foundational for enabling the pipeline to learn from past attempts and improve its strategies over time.

## Running KernelBench
```bash
# evaluate *all* level-1 problems
python run_kernelbench.py --level 1

# run a single problem file
python run_kernelbench.py --problem examples/level1/my_problem.json
```

By default results are printed to stdout and written under
`KernelBench/results/…`.

## End-to-end example
```bash
python examples/simple_bmm.py   # small demo that triggers the pipeline
```
Expected output (abridged):
```
[Synthesis] produced 218-line kernel
[Compile   ] PTX length = 12 kb (target sm_90)
[Benchmark ] 5.2 µs / iter  (1.8× faster than Torch)
✅ correct (max diff 2e-5)
```

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `Assertion 'Index < size()' failed` inside LLVM | The ASTSource signature is inconsistent with kernel parameters – ensure compile-time constants occupy unique slots. |
| `triton.compile() got unexpected keyword` | Multiple Triton versions in env – `pip uninstall triton` until only one remains. |
| GenAI safety-filter runtime error | We now log and return an empty string; pipeline continues but you may get worse hints. |
| `KeyError: 'achieved_flops_gflops'` in performance reports | Fixed: Key name inconsistency in report generation – now uses consistent `achieved_performance_gflops` key. |
| `Session has not been initialized: 0` during Proton profiling | Fixed: Removed premature session deactivation in Proton profiler context manager. |
| `TypeError: 'NoneType' object is not a mapping` in pipeline orchestrator | Fixed: Added comprehensive null checks and defensive programming in `_prepare_final_result` method to handle edge cases where `best_result_data` might be corrupted or None. The pipeline now safely handles dictionary unpacking operations and provides detailed error logging. |

Please file issues / PRs if you hit other rough edges! 
