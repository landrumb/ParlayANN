#!/bin/bash

# check for input argument
if [ "$#" -ne 1 ]; then
    echo "usage: $0 <slice_size>"
    exit 1
fi

# input argument
slice_size=$1

# paths and configurations
DATA_DIR="/pscratch/sd/l/landrum/data/gist/slices"
QUERY_PATH="${DATA_DIR}/query_10000.fbin"
GT_PATH="${DATA_DIR}/GT/test.gt"
EXECUTABLE="./compute_groundtruth"

# check if the base file exists
BASE_PATH="${DATA_DIR}/base_${slice_size}.fbin"
if [ ! -f "$BASE_PATH" ]; then
    echo "error: base file $BASE_PATH does not exist."
    exit 1
fi

# format the slice size with commas
formatted_size=$(printf "%'d" "$slice_size")

# run the executable
echo "Running groundtruth computation for slice size ${formatted_size}"
start_time=$(date +%s.%N)

$EXECUTABLE \
    -base_path "${BASE_PATH}" \
    -query_path "${QUERY_PATH}" \
    -data_type float \
    -dist_func Euclidian \
    -k 100 \
    -gt_path "${GT_PATH}"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
formatted_time=$(printf "%'.3f" "$elapsed")
echo "Execution time: ${formatted_time} seconds"
