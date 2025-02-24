#!/bin/bash
# Iterate over nodes in powers of 2 from 1 to 128
# 1 2 4 8 16 

for nodes in 64 128; do
  echo "Submitting jobs with $nodes node(s)..."
  # Submit 10 jobs for each node count
  for run in {1..10}; do
    for method in nccl_nccl mpi_nccl mpi_mpi; do
      echo "Run $run: sbatch -N $nodes fr_run.sh $method"
      sbatch -N "$nodes" fr_run.sh $method
    done
  done
done
