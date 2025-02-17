from mpi4py import MPI
import torch
import torch.distributed as dist
import numpy as np
import os
from uni_dist import ProcessGroups, all_gather_2D, _all_gather
from utils import time_something, init, allclose
from argparse import ArgumentParser

if __name__ == "__main__":
    init()
    parser = ArgumentParser()
    parser.add_argument("--num-gpus-per-node", 
                        type=int, 
                        required=True, 
                        help="specify number of GPUs/GCDs per node")
    args = parser.parse_args()
    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) 
    unit = "MB"

    hybrid_pgs = {}
    for inner_pg in ["nccl", "mpi"]:
        for outer_pg in ["mpi", "nccl"]:
            hybrid_pg = ProcessGroups(args.num_gpus_per_node, 
                                  dist.get_world_size() // args.num_gpus_per_node, 
                                  inner_pg,
                                  outer_pg)
            hybrid_pgs[f'inner={inner_pg}_outer={outer_pg}'] = hybrid_pg

    for size in sizes:
        if dist.get_rank() == 0:
            print(f"output size = {size} {unit}")
        mult = 2**20 if unit == "MB" else 2**10
        output_buffer_numel = size * (mult) // 2 
        input_buffer_numel = output_buffer_numel // dist.get_world_size()

        output_tensor = torch.empty((output_buffer_numel,), dtype=torch.bfloat16, device="cuda")
        input_tensor = torch.empty((input_buffer_numel,), dtype=torch.bfloat16, device="cuda")

        # mpi 
        time_mpi = time_something(_all_gather, output_tensor, input_tensor, group=MPI.COMM_WORLD)

        # nccl
        time_nccl = time_something(_all_gather, output_tensor, input_tensor, group=None)

        output_tensor_gold = output_tensor.clone()
        output_tensor.zero_()

        # hybrid times
        hybrid_times = {}
        for group_type, group in hybrid_pgs.items():
            hybrid_times[group_type] = time_something(all_gather_2D, 
                                                     output_tensor, 
                                                     input_tensor, 
                                                     group=group)
            assert allclose(output_tensor, output_tensor_gold), f"Error in {group_type}"

        if dist.get_rank() == 0:
            print(f"time_mpi = {time_mpi:.2f} ms")
            print(f"time_nccl = {time_nccl:.2f} ms")
            for group_type, time in hybrid_times.items():
                print(f"time_hybrid {group_type}= {time:.2f} ms")
            

        if dist.get_rank() == 0:
            print("===============================")
        
    dist.destroy_process_group()
