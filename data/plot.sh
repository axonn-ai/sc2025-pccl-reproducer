#!/bin/bash

sizes=(16 32 64 128 256 512 1024)
impls=("nccl" "hybrid")

for size in "${sizes[@]}"; do
    for impl in "${impls[@]}"; do
        python plot_eager_vs_rdvz.py "$impl" "$size" 
    done
done