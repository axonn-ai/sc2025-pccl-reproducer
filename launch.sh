#!/bin/bash
# Iterate over nodes in powers of 2 from 1 to 128
# 1 2 4 8 16 
#methods - nccl, mpi, nccl_mpi, mpidirect, nccl_nccl, mpi_mpi, mpi_nccl,  

exec="benchmark_all_gather.py"
method="nccl"

for nodes in 256 512; do
  echo "Submitting jobs with $nodes node(s)..."
  # Submit 10 jobs for each node count
  for run in {1..3}; do
    echo "Run $run: sbatch -N $nodes pm_run.sh $exec $method"
    sbatch -N "$nodes" pm_run.sh $method
  done
done
