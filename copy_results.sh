#!/bin/bash

# Script to copy timestamped results from pod to local folder
# Usage: ./copy_results.sh <pod_name> <timestamp>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <pod_name> <timestamp>"
    echo "Example: $0 rnachat-enhanced-baselines-job-abc123 20241020_143022"
    exit 1
fi

POD_NAME=$1
TIMESTAMP=$2
LOCAL_RESULTS_DIR="./result/enhanced_${TIMESTAMP}"

echo "Copying results from pod $POD_NAME with timestamp $TIMESTAMP..."

# Create local directory
mkdir -p "$LOCAL_RESULTS_DIR"

# Copy timestamped results folder
echo "Copying results_${TIMESTAMP} folder..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/results_${TIMESTAMP}" "$LOCAL_RESULTS_DIR/"

# Also copy the main results folder as backup
echo "Copying main results folder as backup..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/results" "$LOCAL_RESULTS_DIR/results_backup/"

# Copy training log
echo "Copying training log..."
kubectl cp "${POD_NAME}:/RNAChat_Baselines/training.log" "$LOCAL_RESULTS_DIR/"

echo "Results copied to: $LOCAL_RESULTS_DIR"
echo "Contents:"
ls -la "$LOCAL_RESULTS_DIR"
