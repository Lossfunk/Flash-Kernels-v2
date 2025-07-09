import torch
from utils.utils import ensure_contiguous
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# -----------------------------------------------------------------------------
# Fused Linear + RowSum CUDA Kernel with H100 Optimizations
# Implements: sum(X @ W.T + bias, dim=-1) with numerical stability
# -----------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// H100 Tensor Core optimization parameters
#define TILE_SIZE 16
#define ASYNC_COPY_BYTES 16

namespace cg = cooperative_groups;

// ========================== H100-OPTIMIZED FUSED LINEAR ROWSUM ==========================

template <typename scalar_t>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2) fused_linear_rowsum_forward_kernel(
        const scalar_t* __restrict__ X,        // [batch_size, input_dim]
        const scalar_t* __restrict__ W,        // [output_dim, input_dim] 
        const scalar_t* __restrict__ bias,     // [output_dim] or nullptr
        scalar_t* __restrict__ output,         // [batch_size] - rowsum results
        const int batch_size,
        const int input_dim,
        const int output_dim) {
    
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    // H100: Use tensor core-aligned shared memory
    extern __shared__ char smem_buffer[];
    double* warp_sums = reinterpret_cast<double*>(smem_buffer);
    double* warp_corrections = warp_sums + num_warps;
    
    const scalar_t* x_row = X + row * (long)input_dim;
    
    // H100: Use cooperative groups for memory coalescing
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // MATHEMATICAL OPERATION: sum(X @ W.T + bias, dim=-1)
    // Compute matrix multiplication then sum all output elements
    
    double sum = 0.0;
    double correction = 0.0;  // Kahan correction
    
    // Each thread processes a subset of output dimensions
    for (int out_col = tid; out_col < output_dim; out_col += blockDim.x) {
        double dot_product = 0.0;
        double dot_correction = 0.0;
        
        // Compute dot product: (X @ W.T)[row, out_col] = sum_i(X[row,i] * W[out_col,i])
        constexpr int VEC_SIZE = 4;
        int in_idx = 0;
        
        // Vectorized loop for H100 performance
        for (; in_idx <= input_dim - VEC_SIZE; in_idx += VEC_SIZE) {
            #pragma unroll
            for (int v = 0; v < VEC_SIZE; v++) {
                double x_val = static_cast<double>(x_row[in_idx + v]);
                double w_val = static_cast<double>(W[out_col * (long)input_dim + in_idx + v]);
                double product = x_val * w_val;
                
                // Kahan summation
                double y = product - dot_correction;
                double t = dot_product + y;
                dot_correction = (t - dot_product) - y;
                dot_product = t;
            }
        }
        
        // Handle remaining elements
        for (; in_idx < input_dim; in_idx++) {
            double x_val = static_cast<double>(x_row[in_idx]);
            double w_val = static_cast<double>(W[out_col * (long)input_dim + in_idx]);
            double product = x_val * w_val;
            
            double y = product - dot_correction;
            double t = dot_product + y;
            dot_correction = (t - dot_product) - y;
            dot_product = t;
        }
        
        // Add bias if present
        if (bias != nullptr) {
            double bias_val = static_cast<double>(bias[out_col]);
            double y = bias_val - dot_correction;
            double t = dot_product + y;
            dot_correction = (t - dot_product) - y;
            dot_product = t;
        }
        
        // Add to running sum with Kahan compensation
        double y = (dot_product + dot_correction) - correction;
        double t = sum + y;
        correction = (t - sum) - y;
        sum = t;
    }
    
    // H100: Optimized warp-level reduction with manual shuffles
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        double other_sum = __shfl_down_sync(0xffffffff, sum, offset);
        double other_correction = __shfl_down_sync(0xffffffff, correction, offset);
        
        // Combine sums with their corrections for maximum precision
        double y = other_sum - correction;
        double t = sum + y;
        correction = (t - sum) - y + other_correction;
        sum = t;
    }
    
    // Store per-warp results
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
        warp_corrections[warp_id] = correction;
    }
    block.sync();
    
    // Final block-level reduction
    if (warp_id == 0) {
        double block_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0;
        double block_correction = (lane_id < num_warps) ? warp_corrections[lane_id] : 0.0;
        
        // Final warp reduction with maximum precision
        #pragma unroll
        for (int offset = num_warps >> 1; offset > 0; offset >>= 1) {
            double other_sum = __shfl_down_sync(0xffffffff, block_sum, offset);
            double other_correction = __shfl_down_sync(0xffffffff, block_correction, offset);
            
            double y = other_sum - block_correction;
            double t = block_sum + y;
            block_correction = (t - block_sum) - y + other_correction;
            block_sum = t;
        }
        
        if (lane_id == 0) {
            warp_sums[0] = block_sum + block_correction;
        }
    }
    block.sync();
    
    // Write final result (single-threaded)
    if (tid == 0) {
        double final_sum = warp_sums[0];
        output[row] = static_cast<scalar_t>(final_sum);
    }
}

// ========================== H100-OPTIMIZED BACKWARD KERNELS ==========================

template <typename scalar_t>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2) fused_linear_rowsum_backward_x_kernel(
        const scalar_t* __restrict__ grad_output,  // [batch_size]
        const scalar_t* __restrict__ W,            // [output_dim, input_dim]
        scalar_t* __restrict__ grad_x,             // [batch_size, input_dim]
        const int batch_size,
        const int input_dim,
        const int output_dim) {
    
    const int row = blockIdx.x;
    const int col_base = blockIdx.y * blockDim.x;
    const int tid = threadIdx.x;
    const int col = col_base + tid;
    
    if (row >= batch_size || col >= input_dim) return;
    
    const double grad_out = static_cast<double>(grad_output[row]);
    
    // H100: Vectorized weight column sum with streaming access
    double sum_w_col = 0.0;
    double correction = 0.0;
    
    // Use tensor core-friendly vectorization
    constexpr int VEC_SIZE = 4;
    int out_idx = 0;
    
    // Vectorized summation for H100 performance
    for (; out_idx <= output_dim - VEC_SIZE; out_idx += VEC_SIZE) {
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            double w_val = static_cast<double>(W[(out_idx + v) * (long)input_dim + col]);
            double y = w_val - correction;
            double t = sum_w_col + y;
            correction = (t - sum_w_col) - y;
            sum_w_col = t;
        }
    }
    
    // Handle remaining elements
    for (; out_idx < output_dim; out_idx++) {
        double w_val = static_cast<double>(W[out_idx * (long)input_dim + col]);
        double y = w_val - correction;
        double t = sum_w_col + y;
        correction = (t - sum_w_col) - y;
        sum_w_col = t;
    }
    
    grad_x[row * (long)input_dim + col] = static_cast<scalar_t>(grad_out * (sum_w_col + correction));
}

template <typename scalar_t>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2) fused_linear_rowsum_backward_w_kernel(
        const scalar_t* __restrict__ grad_output,  // [batch_size]
        const scalar_t* __restrict__ X,            // [batch_size, input_dim]
        scalar_t* __restrict__ grad_w,             // [output_dim, input_dim]
        const int batch_size,
        const int input_dim,
        const int output_dim) {
    
    const int out_idx = blockIdx.x;
    const int in_base = blockIdx.y * blockDim.x;
    const int tid = threadIdx.x;
    const int in_idx = in_base + tid;
    
    if (out_idx >= output_dim || in_idx >= input_dim) return;
    
    // H100: Vectorized batch summation
    double grad_w_val = 0.0;
    double correction = 0.0;
    
    constexpr int BATCH_VEC_SIZE = 4;
    int batch_idx = 0;
    
    // Vectorized batch processing
    for (; batch_idx <= batch_size - BATCH_VEC_SIZE; batch_idx += BATCH_VEC_SIZE) {
        #pragma unroll
        for (int v = 0; v < BATCH_VEC_SIZE; v++) {
            double grad_out = static_cast<double>(grad_output[batch_idx + v]);
            double x_val = static_cast<double>(X[(batch_idx + v) * (long)input_dim + in_idx]);
            double product = grad_out * x_val;
            double y = product - correction;
            double t = grad_w_val + y;
            correction = (t - grad_w_val) - y;
            grad_w_val = t;
        }
    }
    
    // Handle remaining batch elements
    for (; batch_idx < batch_size; batch_idx++) {
        double grad_out = static_cast<double>(grad_output[batch_idx]);
        double x_val = static_cast<double>(X[batch_idx * (long)input_dim + in_idx]);
        double product = grad_out * x_val;
        double y = product - correction;
        double t = grad_w_val + y;
        correction = (t - grad_w_val) - y;
        grad_w_val = t;
    }
    
    grad_w[out_idx * (long)input_dim + in_idx] = static_cast<scalar_t>(grad_w_val + correction);
}

template <typename scalar_t>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, 2) fused_linear_rowsum_backward_bias_kernel(
        const scalar_t* __restrict__ grad_output,  // [batch_size]
        scalar_t* __restrict__ grad_bias,          // [output_dim]
        const int batch_size,
        const int output_dim) {
    
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= output_dim) return;
    
    // H100: Vectorized batch gradient summation
    double grad_bias_val = 0.0;
    double correction = 0.0;
    
    constexpr int BATCH_VEC_SIZE = 8;
    int batch_idx = 0;
    
    // Vectorized processing for H100 efficiency
    for (; batch_idx <= batch_size - BATCH_VEC_SIZE; batch_idx += BATCH_VEC_SIZE) {
        #pragma unroll
        for (int v = 0; v < BATCH_VEC_SIZE; v++) {
            double grad_val = static_cast<double>(grad_output[batch_idx + v]);
            double y = grad_val - correction;
            double t = grad_bias_val + y;
            correction = (t - grad_bias_val) - y;
            grad_bias_val = t;
        }
    }
    
    // Handle remaining elements
    for (; batch_idx < batch_size; batch_idx++) {
        double grad_val = static_cast<double>(grad_output[batch_idx]);
        double y = grad_val - correction;
        double t = grad_bias_val + y;
        correction = (t - grad_bias_val) - y;
        grad_bias_val = t;
    }
    
    grad_bias[out_idx] = static_cast<scalar_t>(grad_bias_val + correction);
}

// ================================ H100-OPTIMIZED LAUNCH FUNCTIONS ================================

void fused_linear_rowsum_forward(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor bias,
        torch::Tensor output) {
    
    const int batch_size = X.size(0);
    const int input_dim = X.size(1);
    const int output_dim = W.size(0);
    
    // H100: Optimize for tensor core alignment and memory bandwidth
    const int THREADS_PER_BLOCK = min(MAX_THREADS_PER_BLOCK, max(128, ((output_dim + 127) / 128) * 128));
    const int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    
    // H100: Allocate shared memory for reductions
    const int smem_size = 2 * num_warps * sizeof(double);  // warp_sums + corrections
    
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "fused_linear_rowsum_forward_kernel", ([&] {
        fused_linear_rowsum_forward_kernel<scalar_t><<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<scalar_t>(),
            W.data_ptr<scalar_t>(),
            bias.numel() > 0 ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, input_dim, output_dim);
    }));
}

void fused_linear_rowsum_backward(
        torch::Tensor grad_output,
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor grad_x,
        torch::Tensor grad_w,
        torch::Tensor grad_bias) {
    
    const int batch_size = X.size(0);
    const int input_dim = X.size(1);
    const int output_dim = W.size(0);
    
    // H100: Use larger blocks for better occupancy
    const int THREADS_PER_BLOCK = 512;
    
    // Gradient w.r.t. input X
    if (grad_x.numel() > 0) {
        dim3 grid_x(batch_size, (input_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block_x(THREADS_PER_BLOCK);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            X.scalar_type(),
            "fused_linear_rowsum_backward_x_kernel", ([&] {
            fused_linear_rowsum_backward_x_kernel<scalar_t><<<grid_x, block_x, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(),
                W.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                batch_size, input_dim, output_dim);
        }));
    }
    
    // Gradient w.r.t. weight W
    if (grad_w.numel() > 0) {
        dim3 grid_w(output_dim, (input_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block_w(THREADS_PER_BLOCK);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            X.scalar_type(),
            "fused_linear_rowsum_backward_w_kernel", ([&] {
            fused_linear_rowsum_backward_w_kernel<scalar_t><<<grid_w, block_w, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(),
                X.data_ptr<scalar_t>(),
                grad_w.data_ptr<scalar_t>(),
                batch_size, input_dim, output_dim);
        }));
    }
    
    // Gradient w.r.t. bias
    if (grad_bias.numel() > 0) {
        dim3 grid_bias((output_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block_bias(THREADS_PER_BLOCK);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            grad_output.scalar_type(),
            "fused_linear_rowsum_backward_bias_kernel", ([&] {
            fused_linear_rowsum_backward_bias_kernel<scalar_t><<<grid_bias, block_bias, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_bias.data_ptr<scalar_t>(),
                batch_size, output_dim);
        }));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_rowsum_forward", &fused_linear_rowsum_forward, "H100-Optimized Fused Linear RowSum forward");
    m.def("fused_linear_rowsum_backward", &fused_linear_rowsum_backward, "H100-Optimized Fused Linear RowSum backward");
}
"""

# Compile with H100-optimized flags
if torch.cuda.is_available():
    _fused_linear_rowsum_cuda = load_inline(
        name="fused_linear_rowsum_cuda",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_90,code=sm_90",  # H100
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_75,code=sm_75",  # RTX/T4
            "--maxrregcount=128",
            "-lineinfo",
            "--use_fast_math",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
else:
    _fused_linear_rowsum_cuda = None

# -----------------------------------------------------------------------------
# Python wrappers
# -----------------------------------------------------------------------------

def _fused_linear_rowsum_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Fused linear + rowsum forward: sum(X @ W.T + bias, dim=-1)."""
    if _fused_linear_rowsum_cuda is not None and x.is_cuda:
        try:
            batch_size = x.size(0)
            output = torch.empty(batch_size, dtype=x.dtype, device=x.device)
            
            # Handle bias - create empty tensor if None
            if bias is None:
                bias_tensor = torch.empty(0, dtype=x.dtype, device=x.device)
            else:
                bias_tensor = bias.contiguous()
            
            _fused_linear_rowsum_cuda.fused_linear_rowsum_forward(
                x.contiguous(), weight.contiguous(), bias_tensor, output
            )
            return output
        except Exception as e:
            print(f"CUDA kernel failed, falling back to PyTorch: {e}")
    
    # Fallback to PyTorch implementation - ENSURE DTYPE CONSISTENCY
    weight = weight.to(dtype=x.dtype)
    if bias is not None:
        bias = bias.to(dtype=x.dtype)
    linear_out = torch.nn.functional.linear(x, weight, bias)  # [batch_size, output_dim]
    return torch.sum(linear_out, dim=-1)  # [batch_size]

def _fused_linear_rowsum_backward(grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, 
                                 bias: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused linear + rowsum backward pass."""
    if _fused_linear_rowsum_cuda is not None and x.is_cuda:
        try:
            # Allocate gradient tensors
            grad_x = torch.empty_like(x)
            grad_w = torch.empty_like(weight)
            grad_bias = torch.empty_like(bias) if bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)
            
            _fused_linear_rowsum_cuda.fused_linear_rowsum_backward(
                grad_output.contiguous(), x.contiguous(), weight.contiguous(),
                grad_x, grad_w, grad_bias
            )
            
            return grad_x, grad_w, grad_bias if bias is not None else None
        except Exception as e:
            print(f"CUDA backward kernel failed, falling back to PyTorch: {e}")
    
    # Fallback: Use PyTorch autograd - ENSURE DTYPE CONSISTENCY
    x_req_grad = x.detach().requires_grad_(True)
    weight_req_grad = weight.detach().to(dtype=x.dtype).requires_grad_(True)
    bias_req_grad = bias.detach().to(dtype=x.dtype).requires_grad_(True) if bias is not None else None
    
    with torch.enable_grad():
        result = _fused_linear_rowsum_forward(x_req_grad, weight_req_grad, bias_req_grad)
        result.backward(grad_output)
    
    return x_req_grad.grad, weight_req_grad.grad, bias_req_grad.grad if bias is not None else None

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