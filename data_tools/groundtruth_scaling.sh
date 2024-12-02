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

make compute_groundtruth

echo "Available cores: $(nproc --all)"
echo "Running on $(hostname)"

echo "starting at $(date)"

DATA_DIR="/pscratch/sd/l/landrum/data/deep1b"
# slice_sizes=( 1000 5000 10000 50000 100000 500000 1000000 )
slice_sizes=(1000 5000 10000 50000 100000 500000 1000000 5000000 10000000 50000000 100000000 500000000 1000000000)

for slice_size in ${slice_sizes[@]}; do
    echo "Computing groundtruth for slice of size $slice_size"
    start_time=$(date +%s.%N)
    ./compute_groundtruth -base_path ${DATA_DIR}/base_${slice_size}.fbin \
        -query_path ${DATA_DIR}/query_10000.fbin -data_type float \
        -dist_func Euclidian -k 100 -gt_path ${DATA_DIR}/GT/test.gt
    end_time=$(date +%s.%N)
    echo $start_time $end_time $slice_size >> groundtruth_benchmarks.txt
done
