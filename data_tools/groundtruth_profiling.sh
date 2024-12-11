#!/bin/bash
#SBATCH --job-name=groundtruth_profiling
#SBATCH --output=groundtruth_profiling_%j.out
#SBATCH --error=groundtruth_profiling_%j.err
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=m4776
#SBATCH -C cpu

cd /global/homes/a/anmol/ParlayANN/data_tools

# https://docs.nersc.gov/tools/performance/craypat/
# Compile with Cray CC for profiling
module load perftools-base perftools
module unload darshan

make CC=CC compute_groundtruth
pat_build compute_groundtruth

echo "Available cores: $(nproc --all)"
echo "Running on $(hostname)"

echo "starting at $(date)"

DATA_DIR="/pscratch/sd/l/landrum/data/gist"

executable="./compute_groundtruth+pat"

# # warmup
# $executable -base_path ${DATA_DIR}/gist_learn.fbin \
#     -query_path ${DATA_DIR}/gist_query.fbin -data_type float \
#     -k 100 -dist_func Euclidian -gt_path ${DATA_DIR}/gist_groundtruth_profiling.ivecs
# echo "ran warmup"

echo "Profiling groundtruth"
start_time=$(date +%s.%N)
$executable -base_path ${DATA_DIR}/gist_learn.fbin \
    -query_path ${DATA_DIR}/gist_query.fbin -data_type float \
    -k 100 -dist_func Euclidian -gt_path ${DATA_DIR}/gist_groundtruth_profiling.ivecs
end_time=$(date +%s.%N)
echo $start_time $end_time "100" "gist" >> groundtruth_profiling.txt
