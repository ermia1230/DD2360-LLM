#!/bin/bash

# Batch script to benchmark and profile attention softmax kernels
# Tests kernels: 1 (baseline), 7 (shared), 8 (parallel), 9 (warp), 10 (vector)
# Generates data similar to matmul benchmark table

set -e

echo "=========================================="
echo "Attention Softmax Kernel Benchmark"
echo "=========================================="
echo ""

# Compile with proper architecture for Tesla T4 (sm_75)
echo "Compiling attention_forward.cu with -arch=sm_75..."
nvcc -O3 --use_fast_math -lcublas -lcublasLt -arch=sm_75 attention_forward.cu -o attention_forward
echo "Compilation successful!"
echo ""

# Array of kernels to test
KERNELS=(1 7 8 9 10)
KERNEL_NAMES=("K1 (Baseline)" "K7 (Shared Mem)" "K8 (Parallel Red)" "K9 (Warp Prims)" "K10 (Vectorized)")

# Create output directory for results
mkdir -p benchmark_results

# File to store results
RESULTS_FILE="benchmark_results/attention_softmax_results.txt"
CSV_FILE="benchmark_results/attention_softmax_results.csv"

# Clear previous results
> $RESULTS_FILE
> $CSV_FILE

# CSV header
echo "Kernel,Execution_Time_ms,Speedup,TFLOPS,Compute_Throughput_%,Memory_Throughput_%" > $CSV_FILE

echo "=========================================="
echo "Benchmarking (Validation happens automatically)"
echo "=========================================="
echo ""
echo "Note: All kernels use block size 256"
echo ""

# Table header
echo "Benchmarking Results:" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE
printf "%-20s | %-20s | %-15s | %-10s | %-20s | %-20s\n" \
    "Kernel" "Execution Time (ms)" "Speedup vs K1" "TFLOPS" "Compute Throughput %" "Memory Throughput %" | tee -a $RESULTS_FILE
echo "--------------------|----------------------|-----------------|------------|----------------------|----------------------" | tee -a $RESULTS_FILE

BASELINE_TIME=""

# Enable quick benchmarking mode (1 iterations instead of 100)
export QUICK_BENCH=1

for i in "${!KERNELS[@]}"; do
    KERNEL=${KERNELS[$i]}
    NAME=${KERNEL_NAMES[$i]}
    
    # Run the kernel and capture benchmark output (suppress stdout, keep stderr for errors)
    OUTPUT=$(./attention_forward $KERNEL 2>&1 | grep -E "(block_size|time)")
    
    # Extract execution time
    # Pattern: "block_size 256 | time Y.YYY ms"
    EXEC_TIME=$(echo "$OUTPUT" | grep "block_size" | grep "time" | awk '{print $5}')
    
    if [ -z "$EXEC_TIME" ]; then
        echo "  Warning: Could not extract timing from output"
        EXEC_TIME="N/A"
    fi
    
    # Profile with ncu (suppress verbose output)
    NCU_OUTPUT=$(ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
                     --csv ./attention_forward $KERNEL 2>&1 | grep -E "^\"")
    
    # Extract metrics from ncu output
    if [ -z "$NCU_OUTPUT" ]; then
        COMPUTE_THROUGHPUT="N/A"
        MEMORY_THROUGHPUT="N/A"
    else
        # Filter for softmax kernel lines only, then extract the last value (the metric)
        COMPUTE_THROUGHPUT=$(echo "$NCU_OUTPUT" | grep -i "softmax" | grep -i "sm__throughput" | grep -oP '"\K[0-9.]+(?=")' | tail -1)
        MEMORY_THROUGHPUT=$(echo "$NCU_OUTPUT" | grep -i "softmax" | grep -i "dram__throughput" | grep -oP '"\K[0-9.]+(?=")' | tail -1)
        
        [ -z "$COMPUTE_THROUGHPUT" ] && COMPUTE_THROUGHPUT="N/A"
        [ -z "$MEMORY_THROUGHPUT" ] && MEMORY_THROUGHPUT="N/A"
    fi
    
    # Calculate TFLOPS (approximate based on problem size)
    # For attention: B=8, T=1024, C=768, NH=12, HS=64
    # QK^T matmul: 2 * B * NH * T * T * HS FLOPs
    # Softmax: ~5 * B * NH * T * T FLOPs (max, sub, exp, sum, div)
    # AttV matmul: 2 * B * NH * T * T * HS FLOPs
    # Total ≈ 4 * B * NH * T * T * HS + 5 * B * NH * T * T
    B=8; T=1024; NH=12; HS=64
    FLOPS=$(echo "scale=2; (4 * $B * $NH * $T * $T * $HS + 5 * $B * $NH * $T * $T) / 1000000000" | bc)
    
    if [ "$EXEC_TIME" != "N/A" ]; then
        TFLOPS=$(echo "scale=3; $FLOPS / ($EXEC_TIME / 1000)" | bc)
    else
        TFLOPS="N/A"
    fi
    
    # Calculate speedup
    if [ $i -eq 0 ]; then
        BASELINE_TIME=$EXEC_TIME
        SPEEDUP="1.00"
    else
        if [ "$EXEC_TIME" != "N/A" ] && [ "$BASELINE_TIME" != "N/A" ]; then
            SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $EXEC_TIME" | bc)
        else
            SPEEDUP="N/A"
        fi
    fi
    
    # Print row
    printf "%-20s | %18s ms | %13sx | %8s | %18s %% | %18s %%\n" \
        "$NAME" "$EXEC_TIME" "$SPEEDUP" "$TFLOPS" "$COMPUTE_THROUGHPUT" "$MEMORY_THROUGHPUT" | tee -a $RESULTS_FILE
    
    # Write to CSV
    echo "$NAME,$EXEC_TIME,$SPEEDUP,$TFLOPS,$COMPUTE_THROUGHPUT,$MEMORY_THROUGHPUT" >> $CSV_FILE
done

echo "" | tee -a $RESULTS_FILE
echo "Results saved to:" | tee -a $RESULTS_FILE
echo "  - $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "  - $CSV_FILE" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

