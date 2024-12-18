#!/bin/bash

# paths and configurations
EXECUTABLE="../compute_groundtruth"

# Define an array of sizes
sizes=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000)

# Data directory and other arguments
DATA_DIR="/ssd1/anndata/gist1m"
QUERY_FILE="${DATA_DIR}/gist_query.fbin"
OUTPUT="/dev/null"

# Log file to store performance data
LOG_FILE="our_log.txt"
echo "Size,Time" > "$LOG_FILE"

# Iterate over the sizes
for n in "${sizes[@]}"; do
    # Generate the dataset file name
    DATASET_FILE="${DATA_DIR}/gist_base_${n}.fbin"
    GT_PATH="${OUTPUT}"

    echo "Running groundtruth computation for dataset size ${n}..."
    echo "- BASE_PATH: ${DATASET_FILE}"
    echo "- QUERY_PATH: ${QUERY_FILE}"
    echo "- GT_PATH: ${GT_PATH}"

    # Check if the dataset file exists
    if [ ! -f "$DATASET_FILE" ]; then
        echo "Error: dataset file $DATASET_FILE does not exist."
        continue
    fi

    # Time the execution
    START_TIME=$(date +%s.%N)
    "./$EXECUTABLE" \
        -base_path "${DATASET_FILE}" \
        -query_path "${QUERY_FILE}" \
        -data_type float \
        -dist_func Euclidian \
        -k 100 \
        -gt_path "${GT_PATH}"
    END_TIME=$(date +%s.%N)

    # Calculate elapsed time
    ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    FORMATTED_TIME=$(printf "%'.3f" "$ELAPSED_TIME")
    
    # Log the result
    echo "${n},${FORMATTED_TIME}" >> "$LOG_FILE"

    # Optional: Display progress
    echo "Size ${n} completed in ${FORMATTED_TIME}s"
done

echo "Performance data logged in $LOG_FILE"
