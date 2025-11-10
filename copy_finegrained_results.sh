#!/bin/bash

# Script to copy fine-grained benchmarks results from pod to local folder
# Usage: ./copy_finegrained_results.sh <pod_name> <timestamp>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <pod_name> <timestamp>"
    echo "Example: $0 rnachat-finegrained-benchmarks-job-abc123 20241020_143022"
    exit 1
fi

POD_NAME=$1
TIMESTAMP=$2
LOCAL_RESULTS_DIR="./result/finegrained_benchmarks_${TIMESTAMP}"

echo "Copying fine-grained benchmarks results from pod $POD_NAME with timestamp $TIMESTAMP..."

# Create local directory
mkdir -p "$LOCAL_RESULTS_DIR"

# Copy finegrained_benchmarks results folder
echo "Copying results/finegrained_benchmarks folder..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/results/finegrained_benchmarks" "$LOCAL_RESULTS_DIR/finegrained_benchmarks/"

# Also copy the main results folder as backup
echo "Copying main results folder as backup..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/results" "$LOCAL_RESULTS_DIR/results_backup/"

# Copy training log
echo "Copying training log..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/training.log" "$LOCAL_RESULTS_DIR/"

# Copy finegrained_benchmarks.py script if needed
echo "Copying finegrained_benchmarks.py script..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/finegrained_benchmarks.py" "$LOCAL_RESULTS_DIR/" 2>/dev/null || echo "Script not found, skipping..."

echo "Results copied to: $LOCAL_RESULTS_DIR"
echo "Contents:"
ls -la "$LOCAL_RESULTS_DIR"
echo ""
echo "Fine-grained benchmarks results:"
ls -la "$LOCAL_RESULTS_DIR/finegrained_benchmarks/" 2>/dev/null || echo "Fine-grained benchmarks folder not found"

