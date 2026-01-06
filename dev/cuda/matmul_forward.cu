/*
Kernels for matmul forward pass.
It's advised to use OpenMP here because the CPU implementation is fairly slow otherwise

Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

version 1 — naive per-element kernel:
    - Each CUDA thread computes one output element out[bt, oc].
    - Good for functional correctness and parameter sweeps (sqrt_block_size controls blockDim.x/y).
Run with:
OMP_NUM_THREADS=32 ./matmul_forward 1

version 2 — tiled GEMM-style kernel:
    - Thread-blocks cover tiles of size TILE_M x TILE_N.
    - Each thread computes one tile element; loop over K is not tiled with shared memory.
Run with:
OMP_NUM_THREADS=32 ./matmul_forward 2

version 3 — tiled + shared-memory:
    - Loads input (BT x C) and weight tiles into shared memory (TILE_K) for reuse.
    - Coalesced loads into shared memory and per-block reductions for dot-products.
Run with:   
OMP_NUM_THREADS=32 ./matmul_forward 3

version 4 — tiled + shared-memory + loop unrolling + restricted pointers:
    - Same as version 3 but with #pragma unroll on the inner k-loop and restricted pointers.
Run with:
OMP_NUM_THREADS=32 ./matmul_forward 4

version 5 — cuBLAS-based GEMM:
    - Uses cublasGemmEx to compute out[BT x OC] = inp[BT x C] * weight^T[C x OC].
    - We call cuBLAS with transA = CUBLAS_OP_T and transB = CUBLAS_OP_N so the output memory layout matches row-major [BT x OC].
    - After GEMM a small kernel adds bias if provided (out[bt,oc] += bias[oc]).
    - Uses CUBLAS_GEMM_DEFAULT for broad GPU compatibility (e.g., T4).
Run with:
OMP_NUM_THREADS=32 ./matmul_forward 5
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
__global__ void matmul_forward_kernel1(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc * C;
        const float* inp_bt = inp + bt * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val; 
    }
}

#define TILE_M 16
#define TILE_N 16
// kernel 2: naive kernel with tiling and GEMM structure
__global__ void matmul_forward_kernel2(float* out, //output matrix [BT, OC]
                                       const float* inp, // input matrix [BT, C]
                                       const float* weight, // weight matrix [OC, C]
                                       const float* bias, // bias vector [OC]
                                       int BT, int C, int OC) {

    // Thread coordinates in block
    int ty = threadIdx.y; // row within the block
    int tx = threadIdx.x; // column within the block

    // Map threads to matrix rows and columns
    int bt = blockIdx.x * TILE_M + ty; // row index in output
    int oc = blockIdx.y * TILE_N + tx; // column index in output
    
    // check boundarie conditions
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc]: 0.0f; // add bias if provided

        const float* wrow = weight + oc * C; // point to the start of the weight row
        const float* inp_bt = inp + bt * C; // point to the start of the input row

        // compute the dot product
        for (int i = 0; i < C; i++){
            val += inp_bt[i] * wrow[i];
        }
        // write back the result
        out[bt * OC + oc] = val;
    }
}


#define TILE_K 32
// kernel 3: kernel 2 with shared memory and memory coalescing
__global__ void matmul_forward_kernel3(float* out, //output matrix [BT, OC]
                                       const float* inp, // input matrix [BT, C]
                                       const float* weight, // weight matrix [OC, C]
                                       const float* bias, // bias vector [OC]
                                       int BT, int C, int OC) {
     
    // Shared memory
    __shared__ float inp_s[TILE_M][TILE_K + 1]; // input tile
    __shared__ float weight_s[TILE_N][TILE_K + 1]; // weight tile
    
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

        // Load input tile into shared memory with coalesced access
        int inp_elems = TILE_M * TILE_K;
        for (int i = thread_id; i < inp_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.x * TILE_M + row;
            int global_col = k0 + col;
            if (global_row < BT && global_col < C) {
                inp_s[row][col] = inp[global_row * C + global_col];
            } else {
                inp_s[row][col] = 0.0f;
            }
        }

        // Load weight tile into shared memory with coalesced access
        int weight_elems = TILE_N * TILE_K;
        for (int i = thread_id; i < weight_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.y * TILE_N + row;
            int global_col = k0 + col;          
            if (global_row < OC && global_col < C) {
                weight_s[row][col] = weight[global_row * C + global_col];
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

    // boundary check before writing back
    if (bt < BT && oc < OC) {
        val += (bias != NULL) ? bias[oc]: 0.0f; // add bias if provided
        out[bt * OC + oc] = val; // write back the result
    }
}

// kernel 4: kernel 3 with loop unrolling and restrict qualifiers
__global__ void matmul_forward_kernel4(float* __restrict__ out, //output matrix [BT, OC]
                                       const float* __restrict__ inp, // input matrix [BT, C]
                                       const float* __restrict__ weight, // weight matrix [OC, C]
                                       const float* __restrict__ bias, // bias vector [OC]
                                       int BT, int C, int OC) {
     
    // Shared memory
    __shared__ float inp_s[TILE_M][TILE_K + 1]; // input tile
    __shared__ float weight_s[TILE_N][TILE_K + 1]; // weight tile
    
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

        // Load input tile into shared memory with coalesced access
        int inp_elems = TILE_M * TILE_K;
        for (int i = thread_id; i < inp_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.x * TILE_M + row;
            int global_col = k0 + col;
            if (global_row < BT && global_col < C) {
                inp_s[row][col] = inp[global_row * C + global_col];
            } else {
                inp_s[row][col] = 0.0f;
            }
        }

        // Load weight tile into shared memory with coalesced access
        int weight_elems = TILE_N * TILE_K;
        for (int i = thread_id; i < weight_elems; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = blockIdx.y * TILE_N + row;
            int global_col = k0 + col;          
            if (global_row < OC && global_col < C) {
                weight_s[row][col] = weight[global_row * C + global_col];
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

    // boundary check before writing back
    if (bt < BT && oc < OC) {
        val += (bias != NULL) ? bias[oc]: 0.0f; // add bias if provided
        out[bt * OC + oc] = val; // write back the result
    }
}

// helper kernel to add bias after cuBLAS matmul
__global__ void add_bias_kernel(
    float* out,
    const float* bias,
    int BT,
    int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BT * OC) {
        int o = idx % OC;     // output channel
        out[idx] += bias[o];
    }
}

// kernel 5: use cuBLASGemmEx for matmul + helper kernel for bias addition
void matmul_forward5(float* out,
                     const float* inp,
                     const float* weight,
                     const float* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasCheck(
      cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,      // A = weight^T : C x OC
        CUBLAS_OP_N,      // B = inp      : C x BT
        OC,               // m  (rows of A^T)
        BT,               // n  (cols of B)
        C,                // k
        &alpha,
        weight,
        CUDA_R_32F,
        C,                // lda = rows of weight^T
        inp,
        CUDA_R_32F,
        C,                // ldb = rows of inp
        &beta,
        out,
        CUDA_R_32F,
        OC,               // ldc
        cublas_compute_type,
        CUBLAS_GEMM_DEFAULT
     )
    );

    if (bias != NULL) {
        int threads = 256;
        int blocks  = (BT * OC + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(out, bias, BT, OC);
        cudaCheck(cudaGetLastError());
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

// kernel 2 is a naive kernel in GEMM structure with tiling
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(
        ceil_div(BT, TILE_M),
        ceil_div(OC, TILE_N)
    );

    matmul_forward_kernel2<<<gridDim, blockDim>>>(
        out, inp, weight, bias, BT, C, OC
    );
    cudaCheck(cudaGetLastError());
}

// kernel 3 is kernel 2 with shared memory and memory coalescing
void matmul_forward3(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(
        ceil_div(BT, TILE_M),
        ceil_div(OC, TILE_N)
    );

    matmul_forward_kernel3<<<gridDim, blockDim>>>(
        out, inp, weight, bias, BT, C, OC
    );
    cudaCheck(cudaGetLastError());
}

// kernel 4 is kernel 3 with loop unrolling and restrict qualifiers 
void matmul_forward4(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim(
        ceil_div(BT, TILE_M),
        ceil_div(OC, TILE_N)
    );

    matmul_forward_kernel4<<<gridDim, blockDim>>>(
        out, inp, weight, bias, BT, C, OC
    );
    cudaCheck(cudaGetLastError());
}


// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC);
            break;
        case 3:
            matmul_forward3(out, inp, weight, bias, B, T, C, OC);
            break;
        case 4:
            matmul_forward4(out, inp, weight, bias, B, T, C, OC);
            break;
        case 5:
            matmul_forward5(out, inp, weight, bias, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 32;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // time the kernel at different block sizes
    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        validate_result(d_out, out, "out", B * T * OC, 1e-1f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                              kernel_num, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, OC, sqrt_block_size);

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    return 0;
}