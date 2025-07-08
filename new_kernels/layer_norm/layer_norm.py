import torch
from utils.utils import ensure_contiguous
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# -----------------------------------------------------------------------------
# Inline CUDA kernels – warp-level LayerNorm (forward + backward)
# -----------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>  // for at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
// ---------------------------------- utils ----------------------------------
// warp-level reduction
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// --------------------------------- forward ---------------------------------

template <typename scalar_t>
__global__ void layer_norm_forward_kernel(
        const scalar_t * __restrict__ X,
        const scalar_t * __restrict__ gamma,
        const scalar_t * __restrict__ beta,
        scalar_t * __restrict__ Y,
        float * __restrict__ mean,
        float * __restrict__ rstd,
        const int M,
        const float eps) {
    const int row = blockIdx.x;                     // one row per block
    const int tid = threadIdx.x;
    const int lane = tid & 31;                     // warp lane id
    const int warp_id = tid >> 5;                 // warp id inside block

    extern __shared__ float shared[];             // dynamic smem (64 floats)
    float *s_sum  = shared;                       // size: 32
    float *s_sumsq = shared + 32;                 // size: 32

    const int n_warp = blockDim.x >> 5;           // warps per block
    const scalar_t *row_x = X + row * (long)M;
          scalar_t *row_y = Y + row * (long)M;

    // 1) compute mean/var with WARP reduce
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int col = tid; col < M; col += blockDim.x) {
        float v = static_cast<float>(row_x[col]);
        sum   += v;
        sumsq += v * v;
    }
    sum   = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    if (lane == 0) {
        s_sum[warp_id]   = sum;
        s_sumsq[warp_id] = sumsq;
    }
    __syncthreads();

    // final reduction across warps – first 32 threads are enough
    if (warp_id == 0) {
        sum   = (tid < n_warp) ? s_sum[tid]   : 0.f;
        sumsq = (tid < n_warp) ? s_sumsq[tid] : 0.f;
        sum   = warp_reduce_sum(sum);
        sumsq = warp_reduce_sum(sumsq);
        if (tid == 0) {
            const float mu   = sum / M;
            const float var  = sumsq / M - mu * mu;
            const float inv_std = rsqrtf(var + eps);
            mean[row] = mu;
            rstd[row] = inv_std;
            // broadcast via smem for reuse below
            s_sum[0]   = mu;
            s_sumsq[0] = inv_std;
        }
    }
    __syncthreads();

    const float mu     = s_sum[0];
    const float invstd = s_sumsq[0];

    // 2) write out
    for (int col = tid; col < M; col += blockDim.x) {
        float x    = static_cast<float>(row_x[col]);
        float g    = static_cast<float>(gamma[col]);
        float b    = static_cast<float>(beta[col]);
        float norm = (x - mu) * invstd;
        float y    = norm * g + b;
        row_y[col] = static_cast<scalar_t>(y);
    }
}

// --------------------------------- backward --------------------------------

template <typename scalar_t>
__global__ void layer_norm_backward_kernel(
        const scalar_t * __restrict__ dY,
        const scalar_t * __restrict__ X,
        const scalar_t * __restrict__ gamma,
        const float    * __restrict__ mean,
        const float    * __restrict__ rstd,
        scalar_t * __restrict__ dX,
        float    * __restrict__ dGamma,
        float    * __restrict__ dBeta,
        const int M) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    extern __shared__ float shared[];             // 64 floats again
    float *s_ds = shared;                         // sum(dy * gamma)
    float *s_db = shared + 32;                    // sum(dy * gamma * x_hat)

    const scalar_t *row_x  = X  + row * (long)M;
    const scalar_t *row_dy = dY + row * (long)M;
          scalar_t *row_dx = dX + row * (long)M;

    const float mu   = mean[row];
    const float invstd = rstd[row];

    float ds = 0.f;
    float db = 0.f;

    for (int col = tid; col < M; col += blockDim.x) {
        float g   = static_cast<float>(gamma[col]);
        float dy  = static_cast<float>(row_dy[col]);
        float xmu = static_cast<float>(row_x[col]) - mu;
        float xhat = xmu * invstd;
        // Correct accumulation for LayerNorm backward
        ds += dy * g;
        db += dy * g * xhat;
    }

    ds = warp_reduce_sum(ds);
    db = warp_reduce_sum(db);

    if (lane == 0) {
        s_ds[warp_id] = ds;
        s_db[warp_id] = db;
    }
    __syncthreads();

    if (warp_id == 0) {
        ds = (tid < (blockDim.x >> 5)) ? s_ds[tid] : 0.f;
        db = (tid < (blockDim.x >> 5)) ? s_db[tid] : 0.f;
        ds = warp_reduce_sum(ds);
        db = warp_reduce_sum(db);
        if (tid == 0) {
            s_ds[0] = ds;
            s_db[0] = db;
        }
    }
    __syncthreads();

    ds = s_ds[0];
    db = s_db[0];

    const float div = 1.f / M;

    for (int col = tid; col < M; col += blockDim.x) {
        float g     = static_cast<float>(gamma[col]);
        float dy    = static_cast<float>(row_dy[col]);
        float xmu   = static_cast<float>(row_x[col]) - mu;
        float xhat  = xmu * invstd;
        float dx    = invstd * (dy * g - div * ds - xhat * div * db);
        row_dx[col] = static_cast<scalar_t>(dx);

        // accumulate dGamma / dBeta (atomic adds)
        float dgamma_part = dy * xhat;
        float dbeta_part  = dy;
        atomicAdd(dGamma + col, dgamma_part);
        atomicAdd(dBeta  + col, dbeta_part);
    }
}

// ----------------------------- C++ launchers -------------------------------

void layer_norm_forward(
        torch::Tensor X,
        torch::Tensor gamma,
        torch::Tensor beta,
        torch::Tensor Y,
        torch::Tensor mean,
        torch::Tensor rstd,
        const float eps) {
    const int N = X.size(0);
    const int M = X.size(1);
    const int BLOCK = 256;
    const int SHMEM = 64 * sizeof(float); // 2 x 32 floats

    dim3 grid(N);
    dim3 block(BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "layer_norm_forward_kernel", ([&] {
        layer_norm_forward_kernel<scalar_t><<<grid, block, SHMEM, at::cuda::getCurrentCUDAStream()>>>(
            X.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            M,
            eps);
    }));
}

void layer_norm_backward(
        torch::Tensor dY,
        torch::Tensor X,
        torch::Tensor gamma,
        torch::Tensor mean,
        torch::Tensor rstd,
        torch::Tensor dX,
        torch::Tensor dGamma,
        torch::Tensor dBeta) {
    const int N = X.size(0);
    const int M = X.size(1);
    const int BLOCK = 256;
    const int SHMEM = 64 * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "layer_norm_backward_kernel", ([&] {
        layer_norm_backward_kernel<scalar_t><<<grid, block, SHMEM, at::cuda::getCurrentCUDAStream()>>>(
            dY.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            dX.data_ptr<scalar_t>(),
            dGamma.data_ptr<float>(),
            dBeta.data_ptr<float>(),
            M);
    }));
}

// ----------------------------- PyBind bindings -----------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_forward", &layer_norm_forward, "LayerNorm forward (CUDA)");
    m.def("layer_norm_backward", &layer_norm_backward, "LayerNorm backward (CUDA)");
}
"""

# Compile & load the extension exactly once per process
if torch.cuda.is_available():
    # We provide our own pybind11 module inside the CUDA source, so we do NOT
    # ask `load_inline` to auto-generate binding wrappers via the `functions=`
    # argument. Passing an empty list (the default) disables wrapper
    # generation and avoids missing-declaration errors during compilation.
    _layer_norm_cuda = load_inline(
        name="liger_layer_norm_cuda",
        cpp_sources="",  # not needed – everything lives in the CUDA source
        cuda_sources=_CUDA_SRC,
        verbose=False,
    )
else:
    # If CUDA is not available we avoid the expensive compilation step. Code
    # that relies on this kernel should detect the absence of CUDA and use a
    # safe fallback (see Functional/layer_norm.py).
    _layer_norm_cuda = None

# -----------------------------------------------------------------------------
# Python helpers wrapping the compiled kernels
# -----------------------------------------------------------------------------

def _reshape_input(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Flattens all dimensions except the last one – canonical layout for row-wise LN."""
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.contiguous().view(-1, orig_shape[-1])
    return x, orig_shape

def layer_norm_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float):
    x_flat, shape = _reshape_input(x)
    y_flat = torch.empty_like(x_flat)
    mean  = torch.empty(x_flat.size(0), device=x.device, dtype=torch.float32)
    rstd  = torch.empty_like(mean)

    _layer_norm_cuda.layer_norm_forward(x_flat, gamma.contiguous(), beta.contiguous(), y_flat, mean, rstd, float(eps))

    y = y_flat.view(shape)
    return y, x_flat, mean, rstd, None, None

def layer_norm_backward(dy: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor):
    dy_flat, _ = _reshape_input(dy)
    x_flat, _  = _reshape_input(x)
    M = x_flat.size(1)
    dx_flat    = torch.empty_like(x_flat)
    dgamma = torch.zeros(M, device=x.device, dtype=torch.float32)
    dbeta  = torch.zeros(M, device=x.device, dtype=torch.float32)

    _layer_norm_cuda.layer_norm_backward(dy_flat, x_flat, gamma.contiguous(), mean, rstd, dx_flat, dgamma, dbeta)

    dx = dx_flat.view_as(x)
    return dx, dgamma.to(gamma.dtype), dbeta.to(beta.dtype)

# -----------------------------------------------------------------------------
# Autograd interface
# -----------------------------------------------------------------------------

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float):
        Y, X2d, Mean, RSTD, *_ = layer_norm_forward(X, W, B, eps)
        # save context – X2d is flattened representation
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        ctx.eps = eps
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY: torch.Tensor):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None