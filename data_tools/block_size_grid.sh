#!/bin/bash
#SBATCH --job-name=groundtruth_grid
#SBATCH --output=groundtruth_grid_%j.out
#SBATCH --error=groundtruth_grid_%j.err
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=m4776
#SBATCH -C cpu

cd /global/homes/l/landrum/ParlayANN/data_tools

# make compute_groundtruth

echo "Available cores: $(nproc --all)"
echo "Running on $(hostname)"

echo "starting at $(date)"

DATA_DIR="/pscratch/sd/l/landrum/data/gist/slices"

threads=256
slice_size=1000000
query_size=10000

executable="./compute_groundtruth"

block_sizes=(1 5 10 50 100 500 1000 5000 10000)
# block_sizes=(100)

# warmup
time $executable -base_path ${DATA_DIR}/base_${slice_size}.fbin \
        -query_path ${DATA_DIR}/query_10000.fbin -data_type float \
        -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt

echo "ran warmup"

for query_block_size in ${block_sizes[@]}; do
    for data_block_size in ${block_sizes[@]}; do
        echo "Computing groundtruth with query block size $query_block_size, data block size $data_block_size"
        export PARLAY_NUM_THREADS=$threads
        start_time=$(date +%s.%N)
        $executable -base_path ${DATA_DIR}/base_${slice_size}.fbin \
            -query_path ${DATA_DIR}/query_${query_size}.fbin -data_type float \
            -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt \
            -query_block_size $query_block_size \
            -data_block_size $data_block_size
        end_time=$(date +%s.%N)
        echo $start_time $end_time $threads $query_block_size $data_block_size "gist" $slice_size $query_size >> groundtruth_block_grid.txt
    done
done
