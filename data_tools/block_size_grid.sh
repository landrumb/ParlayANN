#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --output=grid_%j.out
#SBATCH --error=grid_%j.err
#SBATCH --time=03:00:00
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=m4646
#SBATCH -C cpu

cd /global/homes/l/landrum/ParlayANN/data_tools

# make compute_groundtruth

echo "Available cores: $(nproc --all)"
echo "Running on $(hostname)"

echo "starting at $(date)"

DATA_DIR="/pscratch/sd/l/landrum/data/gist/slices"

threads=256
slice_size=500000
query_size=10000

executable="./compute_groundtruth_blocked"

# block_sizes=(1 5 10 50 100 500 1000 5000 10000)
# data_block_sizes=( 1000 2500 5000 )
data_block_sizes=( 500 1000 2500 5000 10000 25000 )
query_block_sizes=( 10 100 250 500 )

# warmup
time $executable -base_path ${DATA_DIR}/base_${slice_size}.fbin \
        -query_path ${DATA_DIR}/query_10000.fbin -data_type float \
        -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt

echo "ran warmup"

for query_block_size in ${query_block_sizes[@]}; do
    for data_block_size in ${data_block_sizes[@]}; do
        echo "Computing groundtruth with query block size $query_block_size, data block size $data_block_size" 
        export PARLAY_NUM_THREADS=$threads
        start_time=$(date +%s.%N)
        time $executable -base_path ${DATA_DIR}/base_${slice_size}.fbin \
            -query_path ${DATA_DIR}/query_${query_size}.fbin -data_type float \
            -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt \
            -query_block_size $query_block_size \
            -data_block_size $data_block_size 
        end_time=$(date +%s.%N)
        echo $start_time $end_time $threads $query_block_size $data_block_size "gist" $slice_size $query_size >> groundtruth_block_grid_serial_outer.txt
        # echo $(awk "BEGIN {print $end_time - $start_time}") $query_block_size $data_block_size "inner_serial" >> "experiments.log"
    done
done
