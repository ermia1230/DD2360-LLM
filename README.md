# Compiling the code
For compiling the main code in a T4 GPU (Google Colab's GPU):

```bash
make train_gpt2fp32cu GPU_COMPUTE_CAPABILITY=75
```

For running the code in different GPUs one might need ti change the GPU_COMPUTE_CAPABILITY parameter to match the GPU architecture.

# Running the code
To run the code, one must first download the model weights and the dataset by running the following commands:
```bash
chmod u+x ./dev/download_starter_pack.sh && ./dev/download_starter_pack.sh
```

After that and after compiling the code, one can start the GPT2 training by running:

```bash
./train_gpt2fp32cu
```

## Run experiments
To run the code with different kernels one must change the variables SOFTMAX_FORWARD_KERNEL, SOFTMAX_BACKWARD_KERNEL and MATMUL_FORWARD_KERNEL to the desired kernels. The default is our best version (all set to 5).

Note: To run the code exactly how we did it in the experiments one must change (in lines 2332 and 2333):
```cpp
int train_num_batches = train_loader.num_tokens / (B*T); // let's do 1 epoch by default for now
int val_num_batches = val_loader.num_tokens / (B*T);
```

to

```cpp
int train_num_batches = 1;
int val_num_batches = 1;
```

and (in line 2376)

```cpp
if (step > 0 && step % sample_every == 0 || last_step)
```

to

```cpp
if (0)
```

## Profiling
To know how much time each kernel takes in each version we used:
```bash
nvprof ./train_gpt2fp32cu
```

To profile the kernels we used:
```bash
ncu -o profile_report ./train_gpt2fp32cu  
```

# Checking the outputs
To check the outputs one can use the appropriate files in the dev/cuda/ folder.

## Test softmax_autoregressive_backward_kernel
Compile attention_backward.cu using:
```bash
cd dev/cuda && nvcc -O3 --use_fast_math -Xcompiler -fopenmp -arch=sm_75 attention_backward.cu -o attention_backward -lcublas -lcublasLt
```

Run it with:
```bash
OMP_NUM_THREADS=32 ./attention_backward 1
```

Profiling was done in train_gpt2fp32cu.

Note: One can test kernels o1(2), o2(3) o3(4) and o4(5) by changing the command line argument to 11, 12, 13 and 14, respectively.

## Test matmul_forward_kernel
Compile matmul_forward.cu using:
```bash
cd dev/cuda && nvcc -O3 --use_fast_math -Xcompiler -fopenmp -arch=sm_75 matmul_forward.cu -o matmul_forward -lcublas -lcublasLt
```

Run it with:
```bash
OMP_NUM_THREADS=32 ./matmul_forward 1
```

Profile with:
```bash
ncu --set basic --launch-skip 1 --launch-count 1 ./matmul_forward 1
```

Note: One can test kernels 2, 3, 4, 5 by changing the command line argument to 2, 3, 4, 5, respectively.


## Test softmax_forward_kernel
Run the benchmark script:
```bash
cd dev/cuda && bash benchmark_attention_softmax.sh
```

This will compile and benchmark all softmax kernel optimizations (K1-baseline, K7-shared memory, K8-parallel reduction, K9-warp primitives, K10-vectorized) with automatic validation, execution time measurement, TFLOPS calculation, and NCU profiling for compute/memory throughput.

Results are saved to:
- `benchmark_results/attention_softmax_results.txt`
- `benchmark_results/attention_softmax_results.csv`

Note: K1 (naive baseline) is very slow (~2+ seconds). To skip it, edit the script and change `KERNELS=(1 7 8 9 10)` to `KERNELS=(7 8 9 10)`.

Individual kernel testing:
```bash
cd dev/cuda && nvcc -O3 --use_fast_math -lcublas -lcublasLt -arch=sm_75 attention_forward.cu -o attention_forward
./attention_forward 7  # Test K7 (shared memory)
```

Profile with:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./attention_forward 7
```
