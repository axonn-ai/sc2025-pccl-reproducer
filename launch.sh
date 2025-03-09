#!/bin/bash
# Iterate over nodes in powers of 2 from 1 to 128
# 1 2 4 8 16 
#methods - nccl, mpi, nccl_mpi, mpidirect, nccl_nccl, mpi_mpi, mpi_nccl

for nodes in 1 2 4 8 16 32 64 128 256; do
  echo "Submitting jobs with $nodes node(s)..."
  # Submit 10 jobs for each node count
  for run in {1..10}; do
    for method in mpird; do
      echo "Run $run: sbatch -N $nodes fr_run.sh $method"
      sbatch -N "$nodes" fr_run.sh $method
    done
  done
done
