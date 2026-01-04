/*
GPT-2 Transformer Neural Net trained in raw CUDA
Non-trivial notes to be aware of:

We are being clever in the backward pass to conserve memory.
In particular, all parameters use a += in the backward pass, so we
can later do gradient accumulation. But all activations have = instead of +=
because these are faster (just read, no write). This is okay for all activations
except for those in the residual stream, where the gradients have to add. We make
sure that those parts work out ok and that we do a += as necessary. E.g.,
the layernorms are connected to the residuals so we += in layernorm backward.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


__device__ __forceinline__ void atomicAddHalf(__half* address, __half val) {
    unsigned int* address_as_uint = (unsigned int*)((size_t)address & ~2);
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        __half2 temp = *reinterpret_cast<__half2*>(&old);
        unsigned short index = (size_t)address & 2 ? 1 : 0;
        __half* ptr = reinterpret_cast<__half*>(&temp) + index;
        *ptr = __hadd(*ptr, val);
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&temp));
    } while (assumed != old);
}

// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// Define kernels to use
#define SOFTMAX_FORWARD_KERNEL 5  // 1=baseline, 2=o1(warp), 3=o2(online), 4=o3(vectorized), 5=o4(combined)
#define SOFTMAX_BACKWARD_KERNEL 5


// ----------------------------------------------------------------------------
// CUDA utils

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

namespace cg = cooperative_groups;

// ----------------------------------------------------------------------------
// all the kernels

// naive implementation into kernel, parallelize over B,T, loop over C
__global__ void encoder_forward_kernel1(__half* out,
                               const int* inp, const __half* wte, const __half* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;

    if (idx < N) {
        int b = idx / T;
        int t = idx % T;
        __half* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        const __half* wte_ix = wte + ix * C;
        const __half* wpe_t = wpe + t * C;
        for (int i = 0; i < C; i++) {
            out_bt[i] = __hadd(wte_ix[i], wpe_t[i]);
        }
    }
}

// really bad naive kernel with atomicAdd
__global__ void encoder_backward_kernel1(__half* dwte, __half* dwpe,
                                        const __half* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const __half* dout_btc = dout + b * T * C + t * C + c;
        __half* dwte_ix = dwte + ix * C + c;
        __half* dwpe_tc = dwpe + t * C + c;

        atomicAddHalf(dwte_ix, *dout_btc);
        atomicAddHalf(dwpe_tc, *dout_btc);
    }
}

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
__global__ void layernorm_forward_kernel1(__half* out, float* mean, float* rstd,
                                 const __half* inp, const __half* weight, const __half* bias,
                                 int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const __half* x = inp + idx * C;
        // calculate the mean (in FP32 for numerical stability)
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += __half2float(x[i]);
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = __half2float(x[i]) - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        __half* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (__half2float(x[i]) - m)); // normalized output
            float o = n * __half2float(weight[i]) + __half2float(bias[i]); // scale and shift it
            out_idx[i] = __float2half(o); // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

__global__ void permute_kernel1(__half* q, __half* k, __half* v,
                               const __half* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void permute_kernel_backward(__half* dinp,
                                        const __half* dq, const __half* dk, const __half* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

__global__ void unpermute_kernel1(const __half* inp, __half *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

__global__ void unpermute_kernel_backward(__half* dinp, const __half *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__global__ void scale_kernel(__half* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = __float2half(-INFINITY);
        } else {
            inp[idx] = __hmul(inp[idx], __float2half(scale));
        }
    }
}

// Naive softmax kernel (note: input is already scaled and masked by scale_kernel)
__global__ void softmax_forward_kernel1(__half* out, const __half* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // each thread handles one row
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * T) {
        return;
    }
    
    const __half* inp_row = inp + idx * T;
    __half* out_row = out + idx * T;
    
    // Find max for numerical stability (use FP32 for accumulation)
    float maxval = -INFINITY;
    for (int col = 0; col < T; col++) {
        float val = __half2float(inp_row[col]);
        if (val > maxval) {
            maxval = val;
        }
    }
    
    // Compute exp and sum (use FP32 for accumulation)
    double sum = 0.0;
    for (int col = 0; col < T; col++) {
        float exp_val = expf(__half2float(inp_row[col]) - maxval);
        out_row[col] = __float2half(exp_val);
        sum += exp_val;
    }
    
    // Normalize
    float norm = 1.0f / (float)sum;
    for (int col = 0; col < T; col++) {
        out_row[col] = __hmul(out_row[col], __float2half(norm));
    }
}

// Optimization 1: Add shared memory to cache row data
// Fixes: Poor memory access - Now loads row into shared memory once
__global__ void softmax_forward_kernel_o1(__half* out, const __half* inp, int N, int T) {
    extern __shared__ float shared_row[];
    
    // Each block processes one row
    int row = blockIdx.x;
    if (row >= N * T) {
        return;
    }
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const __half* inp_row = inp + row * T;
    __half* out_row = out + row * T;
    
    // Cooperatively load row into shared memory (coalesced access), converting to FP32
    for (int i = tid; i < T; i += block_size) {
        shared_row[i] = __half2float(inp_row[i]);
    }
    __syncthreads();
    
    // Thread 0 does the computation (still sequential, but from shared memory)
    if (tid == 0) {
        // Find max
        float maxval = -INFINITY;
        for (int col = 0; col < T; col++) {
            if (shared_row[col] > maxval) {
                maxval = shared_row[col];
            }
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int col = 0; col < T; col++) {
            float exp_val = expf(shared_row[col] - maxval);
            shared_row[col] = exp_val;  // Store back to shared memory
            sum += exp_val;
        }
        
        // Normalize
        float norm = 1.0f / sum;
        for (int col = 0; col < T; col++) {
            shared_row[col] *= norm;
        }
    }
    __syncthreads();
    
    // Cooperatively write result back to global memory as FP16
    for (int i = tid; i < T; i += block_size) {
        out_row[i] = __float2half(shared_row[i]);
    }
}

// Optimization 2: Add parallel reduction for max and sum
// Fixes: No parallelism within each row - Now multiple threads work together
__global__ void softmax_forward_kernel_o2(__half* out, const __half* inp, int N, int T) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    if (row >= N * T) {
        return;
    }
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const __half* inp_row = inp + row * T;
    __half* out_row = out + row * T;
    
    // Step 1: Parallel reduction to find max (use FP32 for accumulation)
    float thread_max = -INFINITY;
    for (int i = tid; i < T; i += block_size) {
        thread_max = fmaxf(thread_max, __half2float(inp_row[i]));
    }
    shared[tid] = thread_max;
    __syncthreads();
    
    // Reduce max in shared memory
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float maxval = shared[0];
    __syncthreads();
    
    // Step 2: Parallel computation of exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < T; i += block_size) {
        float exp_val = expf(__half2float(inp_row[i]) - maxval);
        out_row[i] = __float2half(exp_val);
        thread_sum += exp_val;
    }
    shared[tid] = thread_sum;
    __syncthreads();
    
    // Reduce sum in shared memory
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared[0];
    __syncthreads();
    
    // Step 3: Parallel normalization
    float norm = 1.0f / sum;
    for (int i = tid; i < T; i += block_size) {
        out_row[i] = __hmul(out_row[i], __float2half(norm));
    }
}

// Optimization 3: Use warp-level primitives for efficient reductions
// Fixes: No warp-level primitives - Now uses __shfl_down_sync for faster reductions
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel_o3(__half* out, const __half* inp, int N, int T) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    if (row >= N * T) {
        return;
    }
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    int block_size = blockDim.x;
    int warpsPerBlock = block_size / 32;
    
    const __half* inp_row = inp + row * T;
    __half* out_row = out + row * T;
    
    // Step 1: Find max with warp primitives (use FP32 for accumulation)
    float thread_max = -INFINITY;
    for (int i = tid; i < T; i += block_size) {
        thread_max = fmaxf(thread_max, __half2float(inp_row[i]));
    }
    
    // Warp-level reduction
    float warp_max = warpReduceMax(thread_max);
    
    // First thread in each warp writes to shared memory
    if (laneId == 0) {
        shared[warpId] = warp_max;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (tid < warpsPerBlock) {
        warp_max = shared[tid];
    } else {
        warp_max = -INFINITY;
    }
    if (warpId == 0) {
        warp_max = warpReduceMax(warp_max);
    }
    if (tid == 0) {
        shared[0] = warp_max;
    }
    __syncthreads();
    float maxval = shared[0];
    
    // Step 2: Compute exp and sum with warp primitives
    float thread_sum = 0.0f;
    for (int i = tid; i < T; i += block_size) {
        float exp_val = expf(__half2float(inp_row[i]) - maxval);
        out_row[i] = __float2half(exp_val);
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    float warp_sum = warpReduceSum(thread_sum);
    
    if (laneId == 0) {
        shared[warpId] = warp_sum;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (tid < warpsPerBlock) {
        warp_sum = shared[tid];
    } else {
        warp_sum = 0.0f;
    }
    if (warpId == 0) {
        warp_sum = warpReduceSum(warp_sum);
    }
    if (tid == 0) {
        shared[0] = warp_sum;
    }
    __syncthreads();
    float sum = shared[0];
    
    // Step 3: Normalize
    float norm = 1.0f / sum;
    for (int i = tid; i < T; i += block_size) {
        out_row[i] = __hmul(out_row[i], __float2half(norm));
    }
}

// Optimization 4: Improved memory coalescing with float4 loads
// Fixes: Memory coalescing - Uses vectorized loads for better bandwidth
__global__ void softmax_forward_kernel_o4(__half* out, const __half* inp, int N, int T) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    if (row >= N * T) {
        return;
    }
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    int block_size = blockDim.x;
    int warpsPerBlock = block_size / 32;
    
    const __half* inp_row = inp + row * T;
    __half* out_row = out + row * T;
    
    int T_vec = T / 4;  // Number of float4 elements
    
    // Step 1: Find max with vectorized loads (using FP32 for accumulation)
    float thread_max = -INFINITY;
    for (int i = tid; i < T_vec; i += block_size) {
        // Load 4 halves as 2 floats (half2)
        const half2* inp_ptr = reinterpret_cast<const half2*>(&inp_row[i * 4]);
        half2 vals0 = inp_ptr[0];
        half2 vals1 = inp_ptr[1];
        thread_max = fmaxf(thread_max, fmaxf(__half2float(vals0.x), __half2float(vals0.y)));
        thread_max = fmaxf(thread_max, fmaxf(__half2float(vals1.x), __half2float(vals1.y)));
    }
    // Handle remainder
    for (int i = T_vec * 4 + tid; i < T; i += block_size) {
        thread_max = fmaxf(thread_max, __half2float(inp_row[i]));
    }
    
    // Warp reduction
    float warp_max = warpReduceMax(thread_max);
    if (laneId == 0) shared[warpId] = warp_max;
    __syncthreads();
    
    if (tid < warpsPerBlock) warp_max = shared[tid];
    else warp_max = -INFINITY;
    if (warpId == 0) warp_max = warpReduceMax(warp_max);
    if (tid == 0) shared[0] = warp_max;
    __syncthreads();
    float maxval = shared[0];
    
    // Step 2: Compute exp and sum with vectorized operations
    float thread_sum = 0.0f;
    for (int i = tid; i < T_vec; i += block_size) {
        const half2* inp_ptr = reinterpret_cast<const half2*>(&inp_row[i * 4]);
        half2 vals0 = inp_ptr[0];
        half2 vals1 = inp_ptr[1];
        
        float exp_val0 = expf(__half2float(vals0.x) - maxval);
        float exp_val1 = expf(__half2float(vals0.y) - maxval);
        float exp_val2 = expf(__half2float(vals1.x) - maxval);
        float exp_val3 = expf(__half2float(vals1.y) - maxval);
        
        half2* out_ptr = reinterpret_cast<half2*>(&out_row[i * 4]);
        out_ptr[0] = make_half2(__float2half(exp_val0), __float2half(exp_val1));
        out_ptr[1] = make_half2(__float2half(exp_val2), __float2half(exp_val3));
        
        thread_sum += exp_val0 + exp_val1 + exp_val2 + exp_val3;
    }
    // Handle remainder
    for (int i = T_vec * 4 + tid; i < T; i += block_size) {
        float exp_val = expf(__half2float(inp_row[i]) - maxval);
        out_row[i] = __float2half(exp_val);
        thread_sum += exp_val;
    }
    
    // Warp reduction for sum
    float warp_sum = warpReduceSum(thread_sum);
    if (laneId == 0) shared[warpId] = warp_sum;
    __syncthreads();
    
    if (tid < warpsPerBlock) warp_sum = shared[tid];
    else warp_sum = 0.0f;
    if (warpId == 0) warp_sum = warpReduceSum(warp_sum);
    if (tid == 0) shared[0] = warp_sum;
    __syncthreads();
    float sum = shared[0];
    
    // Step 3: Normalize with vectorized stores
    __half norm_h = __float2half(1.0f / sum);
    for (int i = tid; i < T_vec; i += block_size) {
        half2* out_ptr = reinterpret_cast<half2*>(&out_row[i * 4]);
        half2 vals0 = out_ptr[0];
        half2 vals1 = out_ptr[1];
        out_ptr[0] = make_half2(__hmul(vals0.x, norm_h), __hmul(vals0.y, norm_h));
        out_ptr[1] = make_half2(__hmul(vals1.x, norm_h), __hmul(vals1.y, norm_h));
    }
    // Handle remainder
    for (int i = T_vec * 4 + tid; i < T; i += block_size) {
        out_row[i] = __hmul(out_row[i], norm_h);
    }
}

__global__ void residual_forward_kernel1(__half* out, const __half* inp1, const __half* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __hadd(inp1[idx], inp2[idx]);
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel1(__half* out, const __half* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = __half2float(inp[i]);
        float cube = 0.044715f * xi * xi * xi;
        float result = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
        out[i] = __float2half(result);
    }
}

__global__ void gelu_backward_kernel1(__half* dinp, const __half* inp, const __half* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = __half2float(inp[i]);
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = __float2half(local_grad * __half2float(dout[i]));
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
__global__ void matmul_backward_bias_kernel1(__half* dbias, const __half* dout, int B, int T, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const __half* x = dout + o;
    // thread coarsening (use FP32 for accumulation)
    float sum = 0.0f;
    for (int i = tid; i < B * T; i += block_size) {
        sum += __half2float(x[i * OC]);
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = __hadd(dbias[o], __float2half(shared[0]));
    }
}

// super naive kernel that just parallelizes over B,T and loops over C
__global__ void layernorm_backward_kernel1(__half* dinp, __half* dweight, __half* dbias,
                        const __half* dout, const __half* inp, const __half* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const __half* dout_bt = dout + b * T * C + t * C;
    const __half* inp_bt = inp + b * T * C + t * C;
    __half* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // first: two reduce operations (use FP32 for accumulation)
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (__half2float(inp_bt[i]) - mean_bt) * rstd_bt;
        float dnorm_i = __half2float(weight[i]) * __half2float(dout_bt[i]);
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (__half2float(inp_bt[i]) - mean_bt) * rstd_bt;
        float dnorm_i = __half2float(weight[i]) * __half2float(dout_bt[i]);
        // gradient contribution to bias
        atomicAddHalf(&dbias[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAddHalf(&dweight[i], __float2half(norm_bti * __half2float(dout_bt[i])));
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] = __hadd(dinp_bt[i], __float2half(dval));
    }
}

__global__ void softmax_autoregressive_backward_kernel1(__half* dpreatt, const __half* datt, const __half* att,
                                                     int B, int T, int C, int NH) {
    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    if (t3 < T) {
        int hs = C / NH; // head size
        float scale = 1.0f / sqrtf(hs);
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                for (int t = t3; t < T; t++) {
                    const __half* att_bth = att + b*NH*T*T + h*T*T + t*T;
                    const __half* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                    __half* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                    float accum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = __half2float(att_bth[t2]) * (indicator - __half2float(att_bth[t3]));
                        accum +=  scale * local_derivative * __half2float(datt_bth[t2]);
                    }
                    dpreatt_bth[t3] = __float2half(accum);
                }
            }
        }
    }
}

// Paralelize batches and heads
__global__ void softmax_autoregressive_backward_kernel2(__half* dpreatt, const __half* datt, const __half* att,
                                                     int B, int T, int C, int NH) {
    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    if (b < B && h < NH && t3 < T) {
        int hs = C / NH; // head size
        float scale = 1.0f / sqrtf(hs);
        for (int t = t3; t < T; t++) {
            const __half* att_bth = att + b*NH*T*T + h*T*T + t*T;
            const __half* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
            __half* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
            float accum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                float indicator = t2 == t3 ? 1.0f : 0.0f;
                float local_derivative = __half2float(att_bth[t2]) * (indicator - __half2float(att_bth[t3]));
                accum +=  scale * local_derivative * __half2float(datt_bth[t2]);
            }
            dpreatt_bth[t3] = __float2half(accum);
        }
    }
}

// Paralelize sequences
__global__ void softmax_autoregressive_backward_kernel3(__half* dpreatt, const __half* datt, const __half* att,
                                                     int B, int T, int NH, int hs, float scale) {
    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z % NH;
    int b = blockIdx.z / NH;

    if (b < B && h < NH && t3 < T && t>=t3 && t < T) {
        const __half* att_bth = att + b*NH*T*T + h*T*T + t*T;
        const __half* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
        __half* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
        float accum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = __half2float(att_bth[t2]) * (indicator - __half2float(att_bth[t3]));
            accum +=  scale * local_derivative * __half2float(datt_bth[t2]);
        }
        dpreatt_bth[t3] = __float2half(accum);
    }
}

// Implement shared memory
__global__ void softmax_autoregressive_backward_kernel4(__half* dpreatt, const __half* datt, const __half* att,
                                                     int B, int T, int NH, int hs, float scale) {

    extern __shared__ float shared_mem[];
    float* att_shared = shared_mem; // size T
    float* datt_shared = &shared_mem[T]; // size T

    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z % NH;
    int b = blockIdx.z / NH;

    // Fill shared memory (convert to FP32)
    for (int i = threadIdx.x; i <= t; i += blockDim.x) {
        att_shared[i] = __half2float(att[b*NH*T*T + h*T*T + t*T + i]);
        datt_shared[i] = __half2float(datt[b*NH*T*T + h*T*T + t*T + i]);
    }
    __syncthreads();

    if (b < B && h < NH && t3 < T && t>=t3 && t < T) {
        __half* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
        float accum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_shared[t2] * (indicator - att_shared[t3]);
            accum +=  scale * local_derivative * datt_shared[t2];
        }
        dpreatt_bth[t3] = __float2half(accum);
    }
}

// Algorithmic optimization (avoid loop by precomputing dot product and storing in shared memory)
__global__ void softmax_autoregressive_backward_kernel5(__half* dpreatt, const __half* datt, const __half* att,
                                                     int B, int T, int NH, int hs, float scale) {

    extern __shared__ float shared_mem[];
    float* att_shared = shared_mem; // size T
    float* datt_shared = &shared_mem[T]; // size T
    float* partial_dot_product_shared = &shared_mem[2*T]; // size T

    // dpreatt, datt, att are all (B, NH, T, T)
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int h = blockIdx.z % NH;
    int b = blockIdx.z / NH;

    // Fill shared memory (convert to FP32)
    float thread_accum = 0.0f;
    for (int i = threadIdx.x; i <= t; i += blockDim.x) {
        att_shared[i] = __half2float(att[b*NH*T*T + h*T*T + t*T + i]);
        datt_shared[i] = __half2float(datt[b*NH*T*T + h*T*T + t*T + i]);
        thread_accum += att_shared[i] * datt_shared[i];
    }
    partial_dot_product_shared[threadIdx.x] = thread_accum;
    __syncthreads();

    // Reduce to get the full dot product
    int num_active_threads = min(blockDim.x, t + 1);
    while (num_active_threads > 1) {
        int offset = (num_active_threads + 1) / 2;
        if (threadIdx.x < num_active_threads / 2) {
            partial_dot_product_shared[threadIdx.x] += partial_dot_product_shared[threadIdx.x + offset];
        }
        __syncthreads();
        num_active_threads = offset;
    }

    if (b < B && h < NH && t3 < T && t>=t3 && t < T) {
        dpreatt[b*NH*T*T + h*T*T + t*T + t3] = __float2half(scale * att_shared[t3] * (datt_shared[t3] - partial_dot_product_shared[0]));
    }
}

// naive fused kernel
__global__ void adamw_kernel1(__half* params_memory, const __half* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   // Convert gradients to FP32 for optimizer computations
   float grad = __half2float(grads_memory[i]);
   float param = __half2float(params_memory[i]);
   
   // update the first moment (momentum)
   m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grad;
   // update the second moment (RMSprop)
   v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
   float m_hat = m_memory[i] / beta1_correction;
   float v_hat = v_memory[i] / beta2_correction;
   param -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
   params_memory[i] = __float2half(param);
}

struct SoftmaxParams {
    float Scale;
    float Offset;
};

__device__ SoftmaxParams prepare_softmax(cg::thread_block_tile<32>& warp,
                                         int64_t idx, const __half* inp, int V, int P) {
    // this warp (of 32) threads processes one row of inp, i.e. inp[idx, :] of shape (V,)
    // note that inp is actually (B * T, P) but we only use the first V elements
    // this function then calculates:
    // 1) the max value to subtract for numerical stability and
    // 2) the sum normalization factor
    const __half* x = inp + idx * P;
    // thread coarsening loop, where the 32 threads serially process all V elements
    // thread_rank() is in [0, 31], warp.size() is 32
    float maxval = -INFINITY;
    float sumval = 0.0f;
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float v = __half2float(x[i]);
        float old_maxval = maxval;
        // online softmax recurrence from "Online normalizer calculation for softmax" paper
        maxval = fmaxf(maxval, v);
        sumval *= expf((old_maxval - maxval));
        sumval += expf(v - maxval);
    }
    // warp-level reduction to get the maxval across the 32 threads
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    // all 32 threads do a final shift of the sum considering the global max in this row
    sumval *= expf((maxval - global_maxval));
    // warp-level reduction to get the sumval across the 32 threads
    float global_sumval = cg::reduce(warp, sumval, cg::plus<float>{});
    // the final normalization factor
    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}

// same as 2 but not using float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
__global__ void fused_classifier_kernel1(__half* logits, float* losses,
                             const float* dlosses, const int* targets,
                             int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // example: B = 4, T = 1024, block_size = 128 => we'd have grid_size = 1024
    // each block of 4 warps is in charge of 4 rows of the input, one warp per row
    // meta_group_size is the number of warps per block (e.g. 4)
    // meta_group_rank is the index of the warp in the block (e.g. 0, 1, 2, 3)
    int64_t idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) { // there are B * T rows in the input
        return;
    }

    // calculate the offset (maxval) and scale (sumval) for the softmax
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P);

    // in each row (handled by one warp), thread 0 calculates the loss
    // calculate the probability needed for the loss and update losses
    if(warp.thread_rank() == 0) {
        int ix = targets[idx];
        float prob = expf(__half2float(logits[idx * P + ix]) - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // finally all threads calculate the gradients
    // prob is only materialized here temporarily and in registers, never
    // as a full tensor that gets written to global memory
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    int ix = targets[idx];
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float prob = expf(__half2float(logits[idx * P + i]) - sp.Offset) * sp.Scale;
        float indicator = i == ix ? 1.0f : 0.0f;
        logits[idx * P + i] = __float2half((prob - indicator) * dloss);
    }
}

__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

__global__ void matmul_forward_kernel1(__half* out,
                                       const __half* inp, const __half* weight, const __half* bias,
                                       int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? __half2float(bias[oc]) : 0.0f;
        const __half* wrow = weight + oc*C;
        const __half* inp_bt = inp + bt*C;
        for (int i = 0; i < C; i++) {
            val += __half2float(inp_bt[i]) * __half2float(wrow[i]);
        }
        out[bt * OC + oc] = __float2half(val);
    }
}

// kernel 5: naive kernel with tilling and GEMM structure
#define TILE_M 16
#define TILE_N 16
__global__ void matmul_forward_kernel5(__half* out, //output matrix [BT, OC]
                                       const __half* inp, // input matrix [BT, C]
                                       const __half* weight, // weight matrix [OC, C]
                                       const __half* bias, // bias vector [OC]
                                       int BT, int C, int OC) {

    // Thread coordinates in block
    int ty = threadIdx.y; // row within the block
    int tx = threadIdx.x; // column within the block

    // Map threads to matrix rows and columns
    int bt = blockIdx.x * TILE_M + ty; // row index in output
    int oc = blockIdx.y * TILE_N + tx; // column index in output
    
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? __half2float(bias[oc]): 0.0f;

        const __half* wrow = weight + oc * C; // point to the start of the weight row
        const __half* inp_bt = inp + bt * C; // point to the start of the input row

        for (int i = 0; i < C; i++){
            val += __half2float(inp_bt[i]) * __half2float(wrow[i]);
        }

        out[bt * OC + oc] = __float2half(val);
    }
}

// kernel 6: kernel 5 with shared memory and memory coalescing
#define TILE_K 32

__global__ void matmul_forward_kernel6(__half* out, //output matrix [BT, OC]
                                       const __half* inp, // input matrix [BT, C]
                                       const __half* weight, // weight matrix [OC, C]
                                       const __half* bias, // bias vector [OC]
                                       int BT, int C, int OC) {
     
    // Shared memory (keep in FP32 for better accumulation)
    __shared__ float inp_s[TILE_M][TILE_K]; // input tile
    __shared__ float weight_s[TILE_N][TILE_K]; // weight tile
    
    // Thread coordinates in block
    int ty = threadIdx.y; // row within the block
    int tx = threadIdx.x; // column within the block

    // Map threads to matrix rows and columns
    int bt = blockIdx.x * TILE_M + ty; // row index in output
    int oc = blockIdx.y * TILE_N + tx; // column index in output

    float val = 0.0f; // accumulator for the dot product
    
    // Total number of threads in the block
    int num_threads = TILE_M * TILE_N;
    // Unique thread ID within the block
    int thread_id = ty * TILE_N + tx;
    
    // Loop over tiles of K dimension
    for (int k0 = 0; k0 < C; k0 += TILE_K) {

        // Load input tile into shared memory with coalesced access (convert to FP32)
        int inp_elems = TILE_M * TILE_K;
        for (int i = thread_id; i < inp_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.x * TILE_M + row;
            int global_col = k0 + col;
            if (global_row < BT && global_col < C) {
                inp_s[row][col] = __half2float(inp[global_row * C + global_col]);
            } else {
                inp_s[row][col] = 0.0f;
            }
        }

        // Load weight tile into shared memory with coalesced access (convert to FP32)
        int weight_elems = TILE_N * TILE_K;
        for (int i = thread_id; i < weight_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.y * TILE_N + row;
            int global_col = k0 + col;          
            if (global_row < OC && global_col < C) {
                weight_s[row][col] = __half2float(weight[global_row * C + global_col]);
            } else {
                weight_s[row][col] = 0.0f;
            }
        }

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Compute partial dot product using the shared memory tile
        for (int k = 0; k < TILE_K; k++) {
            val += inp_s[ty][k] * weight_s[tx][k];
        }

        // Synchronize before loading the next tile
        __syncthreads();

    }

    if (bt < BT && oc < OC) {
        val += (bias != NULL) ? __half2float(bias[oc]): 0.0f;
        out[bt * OC + oc] = __float2half(val);
    }
}

// kernel 7: kernel 6 with loop unrolling
__global__ void matmul_forward_kernel7(__half* out, //output matrix [BT, OC]
                                       const __half* inp, // input matrix [BT, C]
                                       const __half* weight, // weight matrix [OC, C]
                                       const __half* bias, // bias vector [OC]
                                       int BT, int C, int OC) {
     
    // Shared memory (keep in FP32 for better accumulation)
    __shared__ float inp_s[TILE_M][TILE_K]; // input tile
    __shared__ float weight_s[TILE_N][TILE_K]; // weight tile
    
    // Thread coordinates in block
    int ty = threadIdx.y; // row within the block
    int tx = threadIdx.x; // column within the block

    // Map threads to matrix rows and columns
    int bt = blockIdx.x * TILE_M + ty; // row index in output
    int oc = blockIdx.y * TILE_N + tx; // column index in output

    float val = 0.0f; // accumulator for the dot product
    
    // Total number of threads in the block
    int num_threads = TILE_M * TILE_N;
    // Unique thread ID within the block
    int thread_id = ty * TILE_N + tx;
    
    // Loop over tiles of K dimension
    for (int k0 = 0; k0 < C; k0 += TILE_K) {

        // Load input tile into shared memory with coalesced access (convert to FP32)
        int inp_elems = TILE_M * TILE_K;
        for (int i = thread_id; i < inp_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.x * TILE_M + row;
            int global_col = k0 + col;
            if (global_row < BT && global_col < C) {
                inp_s[row][col] = __half2float(inp[global_row * C + global_col]);
            } else {
                inp_s[row][col] = 0.0f;
            }
        }

        // Load weight tile into shared memory with coalesced access (convert to FP32)
        int weight_elems = TILE_N * TILE_K;
        for (int i = thread_id; i < weight_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.y * TILE_N + row;
            int global_col = k0 + col;          
            if (global_row < OC && global_col < C) {
                weight_s[row][col] = __half2float(weight[global_row * C + global_col]);
            } else {
                weight_s[row][col] = 0.0f;
            }
        }

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Compute partial dot product using the shared memory tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            val += inp_s[ty][k] * weight_s[tx][k];
        }

        // Synchronize before loading the next tile
        __syncthreads();

    }

    if (bt < BT && oc < OC) {
        val += (bias != NULL) ? __half2float(bias[oc]): 0.0f;
        out[bt * OC + oc] = __float2half(val);
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(__half* out,
                     const int* inp, const __half* wte, const __half* wpe,
                     int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_backward(__half* dwte, __half* dwpe,
                    const __half* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel1<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(__half* out, float* mean, float* rstd,
                       __half* inp, __half* weight, __half* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel 1 is the most naive matmul kernel
void matmul_forward1(__half* out,
                    const __half* inp, const __half* weight, const __half* bias,
                    int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    const int BT = B * T;
    dim3 gridDim(CEIL_DIV(BT, 16), CEIL_DIV(OC, 16));
    dim3 blockDim(16, 16);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, BT, C, OC);
    cudaCheck(cudaGetLastError());
}

// kernel 5 is a naive kernel with tiling
void matmul_forward5(__half* out,
                     const __half* inp, const __half* weight, const __half* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(
        CEIL_DIV(BT, TILE_M),
        CEIL_DIV(OC, TILE_N)
    );

    matmul_forward_kernel5<<<gridDim, blockDim>>>(
        out, inp, weight, bias, BT, C, OC
    );
    cudaCheck(cudaGetLastError());
}

// kernel 6 is a naive kernel with tiling, shared memory and memory coalescing 
void matmul_forward6(__half* out,
                     const __half* inp, const __half* weight, const __half* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(
        CEIL_DIV(BT, TILE_M),
        CEIL_DIV(OC, TILE_N)
    );

    matmul_forward_kernel7<<<gridDim, blockDim>>>(
        out, inp, weight, bias, BT, C, OC
    );
    cudaCheck(cudaGetLastError());
}

// kernel 7 is kernel 6 with loop unrolling (identical launcher to 7)
void matmul_forward7(__half* out,
                     const __half* inp, const __half* weight, const __half* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(
        CEIL_DIV(BT, TILE_M),
        CEIL_DIV(OC, TILE_N)
    );

    matmul_forward_kernel7<<<gridDim, blockDim>>>(
        out, inp, weight, bias, BT, C, OC
    );
    cudaCheck(cudaGetLastError());
}

void attention_forward(__half* out, __half* qkvr, __half* att,
                       __half* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    __half *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel1<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // batched matrix multiply with cuBLAS (using GemmEx for tensor core acceleration)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    __half* preatt = inp;
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, 
        CUBLAS_OP_T, CUBLAS_OP_N, 
        T, T, HS, 
        &alpha, 
        k, CUDA_R_16F, HS, T * HS, 
        q, CUDA_R_16F, HS, T * HS, 
        &beta, 
        preatt, CUDA_R_16F, T, T * T, 
        B * NH, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // multiply all elements of preatt elementwise by scale and apply autoregressive mask
    float scale = 1.0 / sqrtf(HS);
    int scale_grid_size = CEIL_DIV(B * NH * T * T, block_size);
    scale_kernel<<<scale_grid_size, block_size>>>(preatt, scale, B, NH, T);
    cudaCheck(cudaGetLastError());
    
    // apply softmax with selected kernel
#if SOFTMAX_FORWARD_KERNEL == 1
    // Baseline: one thread per row, sequential processing
    int grid_size = CEIL_DIV(B * NH * T, softmax_block_size);
    softmax_forward_kernel1<<<grid_size, softmax_block_size>>>(att, preatt, B * NH, T);
#elif SOFTMAX_FORWARD_KERNEL == 2
    // O1: shared memory caching
    int grid_size = B * NH * T;  // One block per row
    size_t shared_mem = T * sizeof(float);
    softmax_forward_kernel_o1<<<grid_size, softmax_block_size, shared_mem>>>(att, preatt, B * NH, T);
#elif SOFTMAX_FORWARD_KERNEL == 3
    // O2: parallel reduction with shared memory
    int grid_size = B * NH * T;  // One block per row
    size_t shared_mem = softmax_block_size * sizeof(float);
    softmax_forward_kernel_o2<<<grid_size, softmax_block_size, shared_mem>>>(att, preatt, B * NH, T);
#elif SOFTMAX_FORWARD_KERNEL == 4
    // O3: warp-level primitives
    int grid_size = B * NH * T;  // One block per row
    size_t shared_mem = (softmax_block_size / 32) * sizeof(float);
    softmax_forward_kernel_o3<<<grid_size, softmax_block_size, shared_mem>>>(att, preatt, B * NH, T);
#elif SOFTMAX_FORWARD_KERNEL == 5
    // O4: vectorized memory access
    int grid_size = B * NH * T;  // One block per row
    size_t shared_mem = (softmax_block_size / 32) * sizeof(float);
    softmax_forward_kernel_o4<<<grid_size, softmax_block_size, shared_mem>>>(att, preatt, B * NH, T);
#endif
    cudaCheck(cudaGetLastError());

    // new approach: first cuBLAS another batched matmul
    __half* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        HS, T, T, 
        &alpha, 
        v, CUDA_R_16F, HS, T * HS, 
        att, CUDA_R_16F, T, T * T, 
        &beta, 
        vaccum, CUDA_R_16F, HS, T * HS, 
        B * NH, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel1<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

void residual_forward(__half* out, __half* inp1, __half* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(__half* out, const __half* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward(__half* dinp, const __half* inp, const __half* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void matmul_backward(__half* dinp, __half* dweight, __half* dbias,
                     __half* dout, __half* inp, __half* weight,
                     int B, int T, int C, int OC) {
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasGemmEx(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        C, B*T, OC, 
        &one, 
        weight, CUDA_R_16F, C, 
        dout, CUDA_R_16F, OC, 
        &zero, 
        dinp, CUDA_R_16F, C, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasGemmEx(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_T, 
        C, OC, B*T, 
        &one, 
        inp, CUDA_R_16F, C, 
        dout, CUDA_R_16F, OC, 
        &one, 
        dweight, CUDA_R_16F, C, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 512;
        const int grid_size = OC; // one block per output channel
        matmul_backward_bias_kernel1<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

void layernorm_backward(__half* dinp, __half* dweight, __half* dbias,
                        const __half* dout, const __half* inp, const  __half* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    layernorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(__half* dinp, __half* dqkvr, __half* dpreatt, __half* datt, __half* scratch,
                        const __half* dout,
                        const __half* qkvr, const __half* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const __half one = __float2half(1.0f);
    const __half zero = __float2half(0.0f); // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const __half *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    __half *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    float one = 1.0f;
    float zero = 0.0f;
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, 
        CUBLAS_OP_T, CUBLAS_OP_N, 
        T, T, HS, 
        &one, 
        v, CUDA_R_16F, HS, T * HS, 
        scratch, CUDA_R_16F, HS, T * HS, 
        &zero, 
        datt, CUDA_R_16F, T, T * T, 
        B * NH, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // backward into dv
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_T, 
        HS, T, T, 
        &one, 
        scratch, CUDA_R_16F, HS, T * HS, 
        att, CUDA_R_16F, T, T * T, 
        &zero, 
        dv, CUDA_R_16F, HS, T * HS, 
        B * NH, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // backward into preatt
    switch (SOFTMAX_BACKWARD_KERNEL)
    {
    case 1: {
        const int softmax_block_size = 256;
        const int softmax_grid_size = CEIL_DIV(T, softmax_block_size);
        softmax_autoregressive_backward_kernel1<<<dim3(softmax_grid_size, 1), softmax_block_size>>>(dpreatt, datt, att, B, T, C, NH);
        break;
    }
    case 2: {
        const int softmax_block_size = 256;
        const dim3 softmax_grid_size(CEIL_DIV(T, softmax_block_size), NH, B);
        softmax_autoregressive_backward_kernel2<<<softmax_grid_size, softmax_block_size>>>(dpreatt, datt, att, B, T, C, NH);
        break;
    }
    case 3: {
        const int softmax_block_size = 256;
        const int hs = C / NH;
        const float scale = 1.0f / sqrtf(hs);
        const dim3 softmax_grid_size(CEIL_DIV(T, softmax_block_size), T, NH * B);
        softmax_autoregressive_backward_kernel3<<<softmax_grid_size, softmax_block_size>>>(dpreatt, datt, att, B, T, NH, hs, scale);
        break;
    }
    case 4: {
        const int softmax_block_size = 256;
        const int hs = C / NH;
        const float scale = 1.0f / sqrtf(hs);
        const dim3 softmax_grid_size(CEIL_DIV(T, softmax_block_size), T, NH * B);
        size_t shared_mem_size = 2 * T * sizeof(float);
        softmax_autoregressive_backward_kernel4<<<softmax_grid_size, softmax_block_size, shared_mem_size>>>(dpreatt, datt, att, B, T, NH, hs, scale);
        break;
    }
    case 5: {
        const int softmax_block_size = 256;
        const int hs = C / NH;
        const float scale = 1.0f / sqrtf(hs);
        const dim3 softmax_grid_size(CEIL_DIV(T, softmax_block_size), T, NH * B);
        size_t shared_mem_size = (2 * T + softmax_block_size) * sizeof(float);
        softmax_autoregressive_backward_kernel5<<<softmax_grid_size, softmax_block_size, shared_mem_size>>>(dpreatt, datt, att, B, T, NH, hs, scale);
        break;
    }
    default: {
        printf("Invalid SOFTMAX_BACKWARD_KERNEL %d\n", SOFTMAX_BACKWARD_KERNEL);
        exit(1);
    }
    }
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        HS, T, T, 
        &one, 
        k, CUDA_R_16F, HS, T * HS, 
        dpreatt, CUDA_R_16F, T, T * T, 
        &zero, 
        dq, CUDA_R_16F, HS, T * HS, 
        B * NH, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // backward into k
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_T, 
        HS, T, T, 
        &one, 
        q, CUDA_R_16F, HS, T * HS, 
        dpreatt, CUDA_R_16F, T, T * T, 
        &zero, 
        dk, CUDA_R_16F, HS, T * HS, 
        B * NH, 
        cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// replaces logits with logit gradients
void fused_classifier3(__half* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 128;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size / 32);
    fused_classifier_kernel1<<<grid_size, block_size>>>(logits, losses, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    __half* wte; // (V, C)
    __half* wpe; // (maxT, C)
    __half* ln1w; // (L, C)
    __half* ln1b; // (L, C)
    __half* qkvw; // (L, 3*C, C)
    __half* qkvb; // (L, 3*C)
    __half* attprojw; // (L, C, C)
    __half* attprojb; // (L, C)
    __half* ln2w; // (L, C)
    __half* ln2b; // (L, C)
    __half* fcw; // (L, 4*C, C)
    __half* fcb; // (L, 4*C)
    __half* fcprojw; // (L, C, 4*C)
    __half* fcprojb; // (L, C)
    __half* lnfw; // (C)
    __half* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    int Vp = config.padded_vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
__half* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, int on_device) {
    // on_device: 0 = CPU, 1 = GPU
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once on the device
    __half* params_memory;
    if (on_device) {
        cudaCheck(cudaMalloc((void**)&params_memory, num_parameters * sizeof(__half)));
    } else {
        params_memory = (__half*)mallocCheck(num_parameters * sizeof(__half));
    }
    // assign all the tensors their place in the array
    __half** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    __half* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    __half* encoded; // (B, T, C)
    __half* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    __half* atty; // (L, B, T, C)
    __half* att; // (L, B, NH, T, T)
    __half* attproj; // (L, B, T, C)
    __half* residual2; // (L, B, T, C)
    __half* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    __half* fch; // (L, B, T, 4*C)
    __half* fch_gelu; // (L, B, T, 4*C)
    __half* fcproj; // (L, B, T, C)
    __half* residual3; // (L, B, T, C)
    __half* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)

    float* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    __half* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    __half* output;
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded (__half)
    act_sizes[1] = L * B * T * C; // ln1 (__half)
    act_sizes[2] = L * B * T; // ln1_mean (float)
    act_sizes[3] = L * B * T; // ln1_rstd (float)
    act_sizes[4] = L * B * T * C; // atty (__half)
    act_sizes[5] = L * B * NH * T * T; // att (__half)
    act_sizes[6] = L * B * T * C; // attproj (__half)
    act_sizes[7] = L * B * T * C; // residual2 (__half)
    act_sizes[8] = L * B * T * C; // ln2 (__half)
    act_sizes[9] = L * B * T; // ln2_mean (float)
    act_sizes[10] = L * B * T; // ln2_rstd (float)
    act_sizes[11] = L * B * T * 4*C; // fch (__half)
    act_sizes[12] = L * B * T * 4*C; // fch_gelu (__half)
    act_sizes[13] = L * B * T * C; // fcproj (__half)
    act_sizes[14] = L * B * T * C; // residual3 (__half)
    act_sizes[15] = B * T * C; // lnf (__half)
    act_sizes[16] = B * T; // lnf_mean (float)
    act_sizes[17] = B * T; // lnf_rstd (float)
    act_sizes[18] = B * T; // losses (float)
    act_sizes[19] = L * B * T * 3*C; // qkvr (__half)
    act_sizes[20] = B * T * max(3*C, max(NH*T, Vp)); // output / scratch (__half)
}

// Backward pass is conceptually quite different from forward, because we can discard
// the activations of a layer as soon as we're done with it. This lets us aggressively
// reuse memory, so that we need far fewer tensors for backward state.
#define NUM_BACKWARD_TENSORS 3
typedef struct {
    __half* bt4c; // (B, T, 4*C)
    __half* preatt; // (B, NH, T, T)
    __half* residual3; // (B, T, C)
} GradActTensors;


void fill_in_grad_act_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C; // bt4c (__half)
    act_sizes[1] = B * NH * T * T; // preatt (__half)
    act_sizes[2] = B * T * C; // residual3 (__half)
}


void* malloc_and_point_mixed(void** targets[], const size_t* act_sizes, const size_t* type_sizes, int n) {
    size_t num_bytes = 0;
    for (size_t i = 0; i < n; i++) {
        num_bytes += act_sizes[i] * type_sizes[i];
    }
    void* acts_memory;
    cudaCheck(cudaMalloc(&acts_memory, num_bytes));
    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = (void*)acts_memory_iterator;
        acts_memory_iterator += act_sizes[i] * type_sizes[i];
    }
    return acts_memory;
}

void* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    void** ptrs[] = {
        (void**)&acts->encoded, (void**)&acts->ln1, (void**)&acts->ln1_mean, (void**)&acts->ln1_rstd, (void**)&acts->atty,
        (void**)&acts->att, (void**)&acts->attproj, (void**)&acts->residual2, (void**)&acts->ln2, (void**)&acts->ln2_mean,
        (void**)&acts->ln2_rstd, (void**)&acts->fch, (void**)&acts->fch_gelu, (void**)&acts->fcproj, (void**)&acts->residual3, (void**)&acts->lnf,
        (void**)&acts->lnf_mean, (void**)&acts->lnf_rstd, (void**)&acts->losses, (void**)&acts->qkvr, (void**)&acts->output
    };
    size_t type_sizes[] = {
        sizeof(__half), sizeof(__half), sizeof(float), sizeof(float), sizeof(__half),
        sizeof(__half), sizeof(__half), sizeof(__half), sizeof(__half), sizeof(float),
        sizeof(float), sizeof(__half), sizeof(__half), sizeof(__half), sizeof(__half), sizeof(__half),
        sizeof(float), sizeof(float), sizeof(float), sizeof(__half), sizeof(__half)
    };
    return malloc_and_point_mixed(ptrs, act_sizes, type_sizes, NUM_ACTIVATION_TENSORS);
}

void* malloc_and_point_backward(GradActTensors* acts, const size_t* act_sizes) {
    void** ptrs[] = {
        (void**)&acts->bt4c, (void**)&acts->preatt, (void**)&acts->residual3
    };
    size_t type_sizes[] = {sizeof(__half), sizeof(__half), sizeof(__half)};
    return malloc_and_point_mixed(ptrs, act_sizes, type_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    __half* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    __half* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    size_t num_activations;
    // gradients of the activations
    GradActTensors grads_acts;
    size_t num_grad_acts;
    void* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) {
        // was bumped from 1 -> 3 to incorporate the padded vocab size
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    // Note: checkpoint file contains FP32, we convert to FP16 during load
    float* params_memory_cpu_fp32 = (float*)mallocCheck(num_parameters * sizeof(float));
    __half* params_memory_cpu_fp16 = (__half*)mallocCheck(num_parameters * sizeof(__half));
    freadCheck(params_memory_cpu_fp32, sizeof(float), num_parameters, model_file);
    // Convert FP32 to FP16 on CPU
    for (size_t i = 0; i < num_parameters; i++) {
        params_memory_cpu_fp16[i] = __float2half(params_memory_cpu_fp32[i]);
    }
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu_fp16, num_parameters * sizeof(__half), cudaMemcpyHostToDevice));
    free(params_memory_cpu_fp32);
    free(params_memory_cpu_fp16);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL) {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    __half* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        __half* l_ln1w = params.ln1w + l * C;
        __half* l_ln1b = params.ln1b + l * C;
        __half* l_qkvw = params.qkvw + l * 3*C * C;
        __half* l_qkvb = params.qkvb + l * 3*C;
        __half* l_attprojw = params.attprojw + l * C * C;
        __half* l_attprojb = params.attprojb + l * C;
        __half* l_ln2w = params.ln2w + l * C;
        __half* l_ln2b = params.ln2b + l * C;
        __half* l_fcw = params.fcw + l * 4*C * C;
        __half* l_fcb = params.fcb + l * 4*C;
        __half* l_fcprojw = params.fcprojw + l * C * 4*C;
        __half* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        __half* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        __half* l_qkvr = acts.qkvr + l * B * T * 3*C;
        __half* l_atty = acts.atty + l * B * T * C;
        __half* l_att = acts.att + l * B * NH * T * T;
        __half* l_attproj = acts.attproj + l * B * T * C;
        __half* l_residual2 = acts.residual2 + l * B * T * C;
        __half* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        __half* l_fch = acts.fch + l * B * T * 4*C;
        __half* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        __half* l_fcproj = acts.fcproj + l * B * T * C;
        __half* l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        __half* scratch = acts.output;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward7(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward7(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward7(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward7(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward7(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->cpu_losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;

    } else {
        // if we don't have targets, we don't have loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if (model->grads_acts_memory != NULL) { cudaCheck(cudaMemset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(__half))); }
    if (model->grads_memory != NULL) { cudaCheck(cudaMemset(model->grads_memory, 0, model->num_parameters * sizeof(__half))); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        // allocate buffers for weight gradients
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(__half)) >> 20);
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass acrtivations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        GPT2Config cfg = model->config;
        cfg.num_layers = 1; // copy the configuration but override number of layers to 1
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, cfg);
        // count up and allocate the space
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) {
            model->num_grad_acts += bw_act_sizes[i];
        }
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(__half)) >> 20);
        // init gradients of parameters and activations to zero
        gpt2_zero_grad(model);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->seq_len;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    // backward the final layernorm
    __half* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    __half* dresidual = grads_acts.residual3; // the main buffer holding the gradient in the backward pass
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        __half* l_ln1w = params.ln1w + l * C;
        __half* l_qkvw = params.qkvw + l * 3*C * C;
        __half* l_attprojw = params.attprojw + l * C * C;
        __half* l_ln2w = params.ln2w + l * C;
        __half* l_fcw = params.fcw + l * 4*C * C;
        __half* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        __half* dl_ln1w = grads.ln1w + l * C;
        __half* dl_ln1b = grads.ln1b + l * C;
        __half* dl_qkvw = grads.qkvw + l * 3*C * C;
        __half* dl_qkvb = grads.qkvb + l * 3*C;
        __half* dl_attprojw = grads.attprojw + l * C * C;
        __half* dl_attprojb = grads.attprojb + l * C;
        __half* dl_ln2w = grads.ln2w + l * C;
        __half* dl_ln2b = grads.ln2b + l * C;
        __half* dl_fcw = grads.fcw + l * 4*C * C;
        __half* dl_fcb = grads.fcb + l * 4*C;
        __half* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        __half* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        __half* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        __half* l_qkvr = acts.qkvr + l * B * T * 3*C;
        __half* l_atty = acts.atty + l * B * T * C;
        __half* l_att = acts.att + l * B * NH * T * T;
        __half* l_residual2 = acts.residual2 + l * B * T * C;
        __half* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        __half* l_fch = acts.fch + l * B * T * 4*C;
        __half* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
        // so we can co-opt it here.
        __half* dl_btc = acts.lnf;
        __half* dl_bt4c = grads_acts.bt4c;
        __half* dl_preatt = grads_acts.preatt;

        // re-use scratch buffer of the forward pass
        __half* scratch = acts.output;

        // backprop this layer
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        // we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        __half* buffer_a = l_atty;
        __half* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need

        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
}

// Gradient clipping kernel - important for FP16 stability
__global__ void gradient_clip_kernel(__half* grads, long num_parameters, float clip_value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters) return;
    
    float grad = __half2float(grads[i]);
    // Clip gradient to prevent overflow in FP16
    grad = fmaxf(-clip_value, fminf(clip_value, grad));
    grads[i] = __float2half(grad);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        cudaCheck(cudaMalloc((void**)&model->m_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->v_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    
    // Clip gradients to prevent overflow/underflow in FP16 (clip to ±10.0)
    gradient_clip_kernel<<<num_blocks, block_size>>>(model->grads_memory, model->num_parameters, 10.0f);
    cudaCheck(cudaGetLastError());
    
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel1<<<num_blocks, block_size>>>(model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
                                              model->num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->grads_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    cudaFreeHost(model->cpu_losses);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler: takes probabilities and samples integers from them

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Logger lite, will probably grow/change some over time

typedef struct {
    FILE *logfile;
    int flush_every; // every how many steps to flush the log
} Logger;

void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}

void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2fp32cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {

    // read in the (optional) command line arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* output_log_file = NULL;
    int B = 4; // batch size
    int T = 1024; // sequence length max
    float learning_rate = 3e-4f;
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50f |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("+-----------------------+----------------------------------------------------+\n");

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // For FP16 operations with tensor cores: use CUBLAS_COMPUTE_32F_FAST_16F
    // This enables FP16 tensor cores with FP32 accumulation for maximum performance
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    // For FP16, we want tensor core acceleration (available on compute capability >= 7.0)
    if (deviceProp.major >= 7) {
        cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_16F;  // Enables FP16 tensor cores
        cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        printf("| device                | %-50s |\n", deviceProp.name);
        printf("| tensor cores          | %-50s |\n", "enabled (FP16)");
    } else {
        cublas_compute_type = CUBLAS_COMPUTE_32F;
        cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
        printf("| device                | %-50s |\n", deviceProp.name);
        printf("| tensor cores          | %-50s |\n", "not available");
    }
    printf("| precision             | %-50s |\n", "FP16 (with FP32 accumulation)");
    printf("+-----------------------+----------------------------------------------------+\n");

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    printf("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf("| channels C            | %-50d |\n", model.config.channels);
    printf("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
    int train_num_batches = train_loader.num_tokens / (B*T); // let's do 1 epoch by default for now
    int val_num_batches = val_loader.num_tokens / (B*T);
    if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
    printf("| train_num_batches     | %-50d |\n", train_num_batches);
    printf("| val_num_batches       | %-50d |\n", val_num_batches);
    printf("+-----------------------+----------------------------------------------------+\n");

    // print model parameter allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf("allocated %d MiB for model parameters\n", (int)round(model.num_parameters * sizeof(__half) / (1024 * 1024)));

    // set up the Logger
    Logger logger;
    logger_init(&logger, output_log_file);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    __half* cpu_logits_half = (__half*)mallocCheck(model.config.vocab_size * sizeof(__half));
    float* cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // train
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    for (int step = 0; step <= train_num_batches; step++) {
        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss
        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % sample_every == 0 || last_step) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                __half* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and convert from FP16 to FP32 for sampling
                cudaCheck(cudaMemcpy(cpu_logits_half, logits, model.config.vocab_size * sizeof(__half), cudaMemcpyDeviceToHost));
                // Convert FP16 to FP32 for sampling
                for (int i = 0; i < model.config.vocab_size; i++) {
                    cpu_logits[i] = __half2float(cpu_logits_half[i]);
                }
                float coin = random_f32(&rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_sum_iteration_time_s += time_elapsed_s;
        int tokens_per_second = (B * T) / time_elapsed_s;
        printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
        logger_log_train(&logger, step, model.mean_loss);
    }
    // add a total average, for optimizations that are only mild improvements
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(cpu_logits_half);
    free(cpu_logits);
    free(gen_tokens);
    cublasCheck(cublasDestroy(cublas_handle));
    logger_free(&logger);

    return 0;
}
#endif