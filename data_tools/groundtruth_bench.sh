#!/bin/bash

#SBATCH --job-name=groundtruth_bench
#SBATCH --output=groundtruth_bench_%j.out
#SBATCH --error=groundtruth_bench_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=m4776

cd /global/homes/l/landrum/ParlayANN/data_tools

make compute_groundtruth

echo "Available cores: $(nproc --all)"
echo "Running on $(hostname)"

start_time=$(date +%s)
echo "starting at $(date)"


time ./compute_groundtruth -base_path ~/data/sift/sift-1M \
    -query_path ~/data/sift/query-10K -data_type uint8 \
    -dist_func Euclidian -k 100 -gt_path ~/data/sift/GT/sift-1M.gt