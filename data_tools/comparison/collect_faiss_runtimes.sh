#!/bin/bash

# Define an array of sizes
sizes=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

# Data directory and other arguments
DATA_DIR="/ssd1/anndata/gist1m"
QUERY_FILE="${DATA_DIR}/gist_query.fbin"
DIMENSION=960
K=100
OUTPUT="/dev/null"

# Log file to store performance data
LOG_FILE="faiss_log.txt"
echo "Size,Time" > "$LOG_FILE"

# Iterate over the sizes
for n in "${sizes[@]}"; do
    # Generate the dataset file name
    DATASET_FILE="${DATA_DIR}/gist_base_${n}.fbin"
    
    # Time the execution
    START_TIME=$(date +%s.%N)
    python faiss_gt.py --dataset "$DATASET_FILE" --queries "$QUERY_FILE" --dimension $DIMENSION --k $K --output $OUTPUT
    END_TIME=$(date +%s.%N)
    
    # Calculate elapsed time
    ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Log the result
    echo "${n},${ELAPSED_TIME}" >> "$LOG_FILE"
    
    # Optional: Display progress
    echo "Size ${n} completed in ${ELAPSED_TIME}s"
done

echo "Performance data logged in $LOG_FILE"
