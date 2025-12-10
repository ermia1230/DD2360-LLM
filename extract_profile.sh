#!/bin/bash

# Script to extract profiling data from ncu-rep file
# Run this on Colab or wherever you have ncu installed

echo "Extracting kernel summary..."
ncu --import "profile_report (2).ncu-rep" --page raw --csv > profile_kernels.csv

echo "Extracting detailed metrics..."
ncu --import "profile_report (2).ncu-rep" --query all > profile_full.txt

echo "Extracting top kernels by duration..."
ncu --import "profile_report (2).ncu-rep" --query-metrics \
  --metrics "regex:.*" \
  --page raw > profile_metrics.txt

echo "Done! Files created:"
echo "  - profile_kernels.csv"
echo "  - profile_full.txt"
echo "  - profile_metrics.txt"
echo ""
echo "Download these files and share them for analysis."
