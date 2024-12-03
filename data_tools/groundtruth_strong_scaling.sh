#!/bin/bash
#SBATCH --job-name=groundtruth_bench
#SBATCH --output=groundtruth_bench_%j.out
#SBATCH --error=groundtruth_bench_%j.err
#SBATCH --time=00:30:00
#SBATCH --qos=debug
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
threads=( 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 )
# threads=( 256 248 240 232 224 216 208 200 192 184 176 168 160 152 144 136 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1 )
slice_size=1000000

executable="./compute_groundtruth_blocked"

# warmup
$executable -base_path ${DATA_DIR}/base_${slice_size}.fbin \
        -query_path ${DATA_DIR}/query_10000.fbin -data_type float \
        -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt

echo "ran warmup"

for t in ${threads[@]}; do
    echo "Computing groundtruth with $t threads"
    export PARLAY_NUM_THREADS=$t
    start_time=$(date +%s.%N)
    $executable -base_path ${DATA_DIR}/base_${slice_size}.fbin \
        -query_path ${DATA_DIR}/query_10000.fbin -data_type float \
        -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt
    end_time=$(date +%s.%N)
    echo $start_time $end_time $t "100" "100" "gist">> groundtruth_strong_scaling.txt
done
