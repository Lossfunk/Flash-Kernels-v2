import torch
from utils.utils import ensure_contiguous
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# -----------------------------------------------------------------------------
# Mathematically Optimal Fused Linear + RowSum Implementation
# Key insight: sum(X @ W.T + bias, dim=-1) = (X @ sum(W, dim=0)) + sum(bias)
# Complexity: O(batch_size * input_dim + output_dim) vs O(batch_size * input_dim * output_dim)
# -----------------------------------------------------------------------------

def _fused_linear_rowsum_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Mathematically optimal fused linear + rowsum forward pass.
    
    Instead of computing sum(X @ W.T + bias, dim=-1) directly, we use:
    sum(X @ W.T + bias, dim=-1) = sum(X @ W.T, dim=-1) + sum(bias)
                                 = X @ sum(W.T, dim=-1) + sum(bias)
                                 = X @ sum(W, dim=0) + sum(bias)
    
    This reduces complexity from O(B*I*O) to O(B*I + O).
    """
    # For numerical stability, use float32 for intermediate computations
    compute_dtype = torch.float32 if x.dtype in [torch.float16, torch.bfloat16] else x.dtype
    
    # Step 1: Compute column sums of weight matrix - O(I*O) -> O(I)
    weight_colsum = torch.sum(weight.to(compute_dtype), dim=0)  # [input_dim]
    
    # Step 2: Matrix-vector multiplication - O(B*I*O) -> O(B*I)  
    result = torch.mv(x.to(compute_dtype), weight_colsum)  # [batch_size]
    
    # Step 3: Add bias sum if present - O(O) -> O(1)
    if bias is not None:
        bias_sum = torch.sum(bias.to(compute_dtype))
        result = result + bias_sum
        
    return result.to(x.dtype)

def _fused_linear_rowsum_backward(grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, 
                                 bias: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for the mathematically optimal implementation.
    
    For the operation: y = sum(X @ W.T + bias, dim=-1)
    
    Gradients:
    - grad_x[b,j] = grad_output[b] * sum_i(W[i,j]) 
    - grad_w[i,j] = sum_b(grad_output[b] * X[b,j]) (same for all i)
    - grad_bias[i] = sum_b(grad_output[b])
    """
    batch_size, input_dim = x.shape
    output_dim = weight.shape[0]
    
    # For numerical stability, use higher precision for intermediate computations
    compute_dtype = torch.float32 if x.dtype in [torch.float16, torch.bfloat16] else x.dtype
    
    # Gradient w.r.t. input: grad_x[b,j] = grad_output[b] * sum_i(W[i,j])
    weight_colsum = torch.sum(weight.to(compute_dtype), dim=0)  # [input_dim]
    grad_x = grad_output.to(compute_dtype).unsqueeze(1) * weight_colsum.unsqueeze(0)  # [batch_size, input_dim]
    grad_x = grad_x.to(x.dtype)
    
    # Gradient w.r.t. weight: grad_w[i,j] = sum_b(grad_output[b] * X[b,j]) for all i
    # This is the same for all output dimensions
    grad_w_row = torch.sum(grad_output.to(compute_dtype).unsqueeze(1) * x.to(compute_dtype), dim=0)  # [input_dim]
    grad_w = grad_w_row.unsqueeze(0).expand(output_dim, -1)  # [output_dim, input_dim]
    grad_w = grad_w.to(weight.dtype)
    
    # Gradient w.r.t. bias: grad_bias[i] = sum_b(grad_output[b]) for all i
    grad_bias = grad_output.to(compute_dtype).sum().expand(output_dim).to(bias.dtype) if bias is not None else None
    
    return grad_x, grad_w, grad_bias

# -----------------------------------------------------------------------------
# Autograd integration
# -----------------------------------------------------------------------------

class FusedLinearRowSumFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
        output = _fused_linear_rowsum_forward(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        grad_x, grad_w, grad_bias = _fused_linear_rowsum_backward(grad_output, x, weight, bias)
        return grad_x, grad_w, grad_bias

# -----------------------------------------------------------------------------
# CUDA kernel as fallback (kept for compatibility but not used by default)
# -----------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

#define WARP_SIZE 32

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void __launch_bounds__(1024, 1) fused_linear_rowsum_kernel(
        const scalar_t* __restrict__ X,
        const scalar_t* __restrict__ W,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        const int batch_size,
        const int input_dim,
        const int output_dim) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float shmem[1024];
    
    const scalar_t* x_row = X + batch_idx * input_dim;
    float thread_sum = 0.0f;
    
    for (int out_start = 0; out_start < output_dim; out_start += blockDim.x) {
        const int out_idx = out_start + tid;
        
        if (out_idx < output_dim) {
            float dot_product = 0.0f;
            const scalar_t* w_row = W + out_idx * input_dim;
            
            for (int k = 0; k < input_dim; k++) {
                dot_product += static_cast<float>(x_row[k]) * static_cast<float>(w_row[k]);
            }
            
            if (bias != nullptr) {
                dot_product += static_cast<float>(bias[out_idx]);
            }
            
            thread_sum += dot_product;
        }
    }
    
    shmem[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = static_cast<scalar_t>(shmem[0]);
    }
}

void fused_linear_rowsum_forward(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor bias,
        torch::Tensor output) {
    
    const int batch_size = X.size(0);
    const int input_dim = X.size(1);
    const int output_dim = W.size(0);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "fused_linear_rowsum_kernel", ([&] {
        
        dim3 grid(batch_size);
        dim3 block(min(1024, max(32, ((output_dim + 31) / 32) * 32)));
        
        fused_linear_rowsum_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            bias.numel() > 0 ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, input_dim, output_dim);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_rowsum_forward", &fused_linear_rowsum_forward, "CUDA Fused Linear RowSum forward");
}
"""

# Compile CUDA kernel (optional fallback)
if torch.cuda.is_available():
    try:
        _fused_linear_rowsum_cuda = load_inline(
            name="fused_linear_rowsum_cuda",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr", 
                "-gencode=arch=compute_90,code=sm_90",
                "-gencode=arch=compute_80,code=sm_80",
                "--maxrregcount=255",
                "-lineinfo",
                "--use_fast_math",
                "--extra-device-vectorization",
                "-Xptxas=-O3",
            ],
            extra_ldflags=["-lcuda", "-lcublas"],
            verbose=False,
        )
    except:
        _fused_linear_rowsum_cuda = None
else:
    _fused_linear_rowsum_cuda = None 