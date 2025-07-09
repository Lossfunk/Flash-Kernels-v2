import torch
from utils.utils import ensure_contiguous
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# -----------------------------------------------------------------------------
# Optimized Diagonal Matrix–Matrix multiply CUDA kernels
# Implements: C[row, col] = A[row] * B[row, col]
# -----------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ------------------------------ FORWARD -------------------------------------

template <typename scalar_t>
__global__ void __launch_bounds__(256, 8) diagonal_matmul_optimized_kernel(
    const scalar_t* __restrict__ A,   // [N]
    const scalar_t* __restrict__ B,   // [N, M]
    scalar_t* __restrict__ C,         // [N, M]
    const int N,
    const int M) {
    
    const int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t a_val = __ldg(A + row);  // cache scalar in register
    const scalar_t* b_row = B + (long)row * M;
    scalar_t* c_row = C + (long)row * M;

    // Vectorized processing - 4 elements per thread
    const int vec_cols = M & ~3;  // multiple of 4
    for (int col4 = threadIdx.x * 4; col4 < vec_cols; col4 += blockDim.x * 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = col4 + i;
            scalar_t b_val = __ldg(b_row + col);
            c_row[col] = a_val * b_val;
        }
    }
    
    // Handle remaining columns
    for (int col = vec_cols + threadIdx.x; col < M; col += blockDim.x) {
        scalar_t b_val = __ldg(b_row + col);
        c_row[col] = a_val * b_val;
    }
}

// ------------------------------ BACKWARD ------------------------------------

template <typename scalar_t>
__global__ void __launch_bounds__(256, 8) diagonal_backward_A_optimized_kernel(
    const scalar_t* __restrict__ grad_output, // [N, M]
    const scalar_t* __restrict__ B,           // [N, M]
    scalar_t* __restrict__ grad_A,            // [N]
    const int N,
    const int M) {
    
    const int row = blockIdx.x;
    if (row >= N) return;

    const scalar_t* go_row = grad_output + (long)row * M;
    const scalar_t* b_row = B + (long)row * M;

    // Use double precision accumulation for better accuracy
    double thread_sum = 0.0;
    for (int col = threadIdx.x; col < M; col += blockDim.x) {
        thread_sum += static_cast<double>(go_row[col]) * static_cast<double>(b_row[col]);
    }

    // Block reduce (sum)
    __shared__ double shmem[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane == 0) {
        shmem[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp aggregates results from all warps in block
    if (warp_id == 0) {
        double block_sum = (lane < (blockDim.x + 31) / 32) ? shmem[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            grad_A[row] = static_cast<scalar_t>(block_sum);
        }
    }
}

// ------------------------------ LAUNCHERS -----------------------------------

void diag_matmul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int N = A.size(0);
    const int M = B.size(1);

    const int THREADS = 256;
    dim3 grid(N);
    dim3 block(THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
                                    A.scalar_type(), "diag_matmul_optimized_forward", ([&] {
        diagonal_matmul_optimized_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), N, M);
    }));
}

void diag_matmul_backward(torch::Tensor grad_output, torch::Tensor A, torch::Tensor B,
                          torch::Tensor grad_A, torch::Tensor grad_B) {
    const int N = A.size(0);
    const int M = B.size(1);

    const int THREADS = 256;
    dim3 grid(N);
    dim3 block(THREADS);

    // grad_A with optimized reduction
    if (grad_A.numel() > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
                                        A.scalar_type(), "diag_matmul_optimized_backward_A", ([&] {
            diagonal_backward_A_optimized_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), grad_A.data_ptr<scalar_t>(), N, M);
        }));
    }

    // grad_B (reuse forward kernel logic)
    if (grad_B.numel() > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
                                        A.scalar_type(), "diag_matmul_optimized_backward_B", ([&] {
            diagonal_matmul_optimized_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                A.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), grad_B.data_ptr<scalar_t>(), N, M);
        }));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diag_matmul_forward", &diag_matmul_forward, "Optimized diagonal matmul forward");
    m.def("diag_matmul_backward", &diag_matmul_backward, "Optimized diagonal matmul backward");
}
"""

# -----------------------------------------------------------------------------
# Compile and load the extension
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
    _diag_matmul_cuda = load_inline(
        name="diag_matmul_optimized_cuda",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr", 
            "-gencode=arch=compute_90,code=sm_90",   # H100 native
            "-gencode=arch=compute_80,code=sm_80",   # A100 fallback
            "-gencode=arch=compute_75,code=sm_75",   # Turing fallback
            "--maxrregcount=128",
            "-lineinfo",
            "--use_fast_math",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
else:
    _diag_matmul_cuda = None

# -----------------------------------------------------------------------------
# Python wrappers
# -----------------------------------------------------------------------------

def _diag_matmul_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Optimized forward: out = diag(a) @ b."""
    assert a.dim() == 1, "a must be 1-D tensor"
    assert b.dim() == 2, "b must be 2-D tensor"
    assert a.size(0) == b.size(0), "a and b must have compatible shapes"

    if _diag_matmul_cuda is not None and a.is_cuda:
        try:
            output = torch.empty_like(b, memory_format=torch.contiguous_format)
            _diag_matmul_cuda.diag_matmul_forward(a.contiguous(), b.contiguous(), output)
            return output
        except Exception as e:
            print(f"CUDA kernel failed, falling back to PyTorch: {e}")

    # fallback
    return a.unsqueeze(-1) * b


def _diag_matmul_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized backward pass."""
    assert grad_output.shape == b.shape, "grad_output shape mismatch"

    if _diag_matmul_cuda is not None and grad_output.is_cuda:
        try:
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            _diag_matmul_cuda.diag_matmul_backward(grad_output.contiguous(), a.contiguous(), b.contiguous(), grad_a, grad_b)
            return grad_a, grad_b
        except Exception as e:
            print(f"CUDA backward kernel failed, falling back to PyTorch: {e}")

    # fallback
    a_req = a.detach().requires_grad_(True)
    b_req = b.detach().requires_grad_(True)
    with torch.enable_grad():
        out = a_req.unsqueeze(-1) * b_req
        out.backward(grad_output)
    return a_req.grad, b_req.grad

# -----------------------------------------------------------------------------
# Autograd integration
# -----------------------------------------------------------------------------

class DiagonalMatMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        output = _diag_matmul_forward(a, b)
        ctx.save_for_backward(a, b)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_a, grad_b = _diag_matmul_backward(grad_output, a, b)
        return grad_a, grad_b 