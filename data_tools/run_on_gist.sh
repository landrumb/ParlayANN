#!/bin/bash

# paths and configurations
EXECUTABLE="compute_groundtruth"
make $EXECUTABLE

# check for input argument
if [ "$#" -gt 1 ]; then
    echo "Usage: $0"
    echo "      $0 <slice_size>"
    exit 1
elif [ "$#" -eq 0 ]; then
    # No input argument, run on full GIST dataset
    DATA_DIR="/pscratch/sd/l/landrum/data/gist"
    BASE_PATH="${DATA_DIR}/gist_learn.fbin"
    QUERY_PATH="${DATA_DIR}/gist_query.fbin"
    GT_PATH="${DATA_DIR}/test.gt"
    echo "Running groundtruth computation for full GIST dataset..."
else
    # Input argument defines slice size, run on GIST slice
    DATA_DIR="/pscratch/sd/l/landrum/data/gist/slices"
    slice_size=$1
    BASE_PATH="${DATA_DIR}/base_${slice_size}.fbin"
    QUERY_PATH="${DATA_DIR}/query_10000.fbin"
    GT_PATH="${DATA_DIR}/GT/test.gt"
    # format the slice size with commas
    formatted_size=$(printf "%'d" "$slice_size")
    # run the executable
    echo "Running groundtruth computation for slice size ${formatted_size}..."
fi
echo "- BASE_PATH: ${BASE_PATH}"
echo "- QUERY_PATH: ${QUERY_PATH}"
echo "- GT_PATH: ${GT_PATH}"

# check if the base file exists
if [ ! -f "$BASE_PATH" ]; then
    echo "error: base file $BASE_PATH does not exist."
    exit 1
fi

start_time=$(date +%s.%N)

"./$EXECUTABLE" \
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
