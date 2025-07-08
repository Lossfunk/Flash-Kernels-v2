"""
Triton Documentation Knowledge Base

This module contains optimized Triton kernel implementations extracted from the official
Triton documentation and tutorials. These serve as high-quality reference implementations
for common operations.

Source: https://triton-lang.org/main/getting-started/tutorials/
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from utils.logging_utils import get_logger
import torch

logger = get_logger("TritonDocsKnowledge")


@dataclass
class TritonDocImplementation:
    """Represents a documented Triton implementation."""
    operation_name: str
    source_url: str
    description: str
    kernel_code: str
    launcher_code: str
    performance_notes: str
    requirements: List[str]


class TritonDocsKnowledgeBase:
    """Knowledge base of optimized Triton implementations from official documentation."""
    
    def __init__(self):
        self._implementations = self._load_implementations()
        self._problem_id_mapping = self._load_problem_id_mapping()
        logger.info(f"Loaded {len(self._implementations)} Triton doc implementations")
        logger.info(f"Loaded {len(self._problem_id_mapping)} problem ID mappings")
    
    def _load_problem_id_mapping(self) -> Dict[int, str]:
        """Map KernelBench problem IDs to operation names."""
        return {
            23: "softmax",  # KernelBench level1/23_Softmax.py
            # Add more mappings as we expand the knowledge base
            # 1: "vector_add",
            # 15: "matmul", 
            # etc.
        }
    
    def _load_implementations(self) -> Dict[str, TritonDocImplementation]:
        """Load all documented implementations."""
        implementations = {}
        
        # Fused Softmax from official Triton tutorial
        implementations["softmax"] = TritonDocImplementation(
            operation_name="softmax",
            source_url="https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html",
            description="Fused softmax operation from 02-fused-softmax.py tutorial. Significantly faster than PyTorch's native op for matrices whose rows can fit in GPU's SRAM.",
            kernel_code='''@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)''',
            launcher_code='''
# Helper functions from 02-fused-softmax.py needed by the launcher
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')

def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y''',
            performance_notes="From 02-fused-softmax.py: Triton is ~4x faster than Torch JIT and noticeably faster than torch.softmax for applicable matrices. Best when rows fit in SRAM.",
            requirements=[
                "Input must be 2D tensor",
                "Rows should fit in GPU SRAM for optimal performance",
                "Uses power-of-2 block sizes for efficiency",
                "Requires triton.runtime.driver for device properties"
            ]
        )
        
        return implementations
    
    def get_implementation(self, operation_name: str) -> Optional[TritonDocImplementation]:
        """Get the documented implementation for an operation."""
        return self._implementations.get(operation_name.lower())
    
    def has_implementation(self, operation_name: str) -> bool:
        """Check if we have a documented implementation for an operation."""
        return operation_name.lower() in self._implementations
    
    def get_operation_from_problem_id(self, problem_id: int) -> Optional[str]:
        """Get the operation name from a KernelBench problem ID."""
        return self._problem_id_mapping.get(problem_id)
    
    def has_problem_id_mapping(self, problem_id: int) -> bool:
        """Check if we have a mapping for a specific problem ID."""
        return problem_id in self._problem_id_mapping
    
    def list_available_operations(self) -> List[str]:
        """List all operations with documented implementations."""
        return list(self._implementations.keys())
    
    def list_mapped_problem_ids(self) -> List[int]:
        """List all problem IDs that have mappings."""
        return list(self._problem_id_mapping.keys())
    
    def detect_operation_from_pytorch(self, pytorch_src: str) -> Optional[str]:
        """Detect the operation type from PyTorch source code."""
        pytorch_src_lower = pytorch_src.lower()
        
        # Simple pattern matching for common operations
        if "torch.softmax" in pytorch_src_lower or "f.softmax" in pytorch_src_lower:
            return "softmax"
        
        # Add more patterns as we expand the knowledge base
        # if "torch.matmul" in pytorch_src_lower or "@" in pytorch_src:
        #     return "matmul"
        # if "torch.relu" in pytorch_src_lower or "f.relu" in pytorch_src_lower:
        #     return "relu"
        
        return None
    
    def detect_operation_from_problem_id(self, problem_id: Optional[int], pytorch_src: str) -> Optional[str]:
        """Detect operation from problem ID first, then fall back to PyTorch source analysis."""
        # First try problem ID mapping
        if problem_id is not None and self.has_problem_id_mapping(problem_id):
            operation = self.get_operation_from_problem_id(problem_id)
            logger.info(f"🎯 PROBLEM ID MAPPING: Problem {problem_id} -> {operation}")
            return operation
        
        # Fall back to PyTorch source analysis
        return self.detect_operation_from_pytorch(pytorch_src)
    
    def generate_optimized_kernel(self, operation_name: str, input_specs: List, device_info: Optional[Dict] = None) -> Optional[str]:
        """Generate a complete optimized kernel based on documented implementation."""
        impl = self.get_implementation(operation_name)
        if not impl:
            return None
        
        logger.info(f"Using optimized Triton docs implementation for {operation_name}")
        logger.info(f"Source: {impl.source_url}")
        
        # Build complete kernel with imports and helper functions - using exact 02-fused-softmax.py pattern
        complete_kernel_parts = ['''import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()''']
        
        # Add kernel code
        complete_kernel_parts.append(impl.kernel_code)
        
        # Add device properties setup (from 02-fused-softmax.py)
        complete_kernel_parts.append('''
# We can create a helper function that enqueues the kernel and its (meta-)arguments for any given input tensor.

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}''')
        
        # Add launcher code (which now includes is_hip, is_cdna)
        complete_kernel_parts.append(impl.launcher_code)

        if operation_name == "softmax":
            # Add unit test and benchmark from 02-fused-softmax.py
            complete_kernel_parts.append('''
# Unit Test
# ---------

# We make sure that we test our kernel on a matrix with an irregular number of rows and columns.
# This will allow us to verify that our padding mechanism works.

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

# As expected, the results are identical.

# Benchmark
# ---------
#
# Here we will benchmark our operation as a function of the number of columns in the input matrix -- assuming 4096 rows.
# We will then compare its performance against (1) :code:`torch.softmax` and (2) the :code:`naive_softmax` defined above.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_softmax(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)''')

        # Add performance notes and requirements as comments
        complete_kernel_parts.append(f'''
# Performance notes from Triton documentation:
# {impl.performance_notes}
#
# Requirements:
# {chr(10).join(f"# - {req}" for req in impl.requirements)}
#
# Source: {impl.source_url}
''')
        complete_kernel = "\n".join(complete_kernel_parts)
        
        return complete_kernel


# Global instance
triton_docs_kb = TritonDocsKnowledgeBase() 