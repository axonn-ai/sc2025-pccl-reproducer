import torch
import torch.distributed as dist
from mpi4py import MPI
import numpy as np
import os
from uni_dist import ProcessGroups, all_gather_2D, _all_gather
from utils import time_something, init, allclose
from argparse import ArgumentParser
import csv

def get_gpu_counts_and_job_id():
    # Get SLURM job details
    gpu_count = int(os.getenv("SLURM_NTASKS", "1"))  # Default to 1 if not found
    slurm_job_id = os.getenv("SLURM_JOB_ID", "unknown")
    return gpu_count, slurm_job_id

if __name__ == "__main__":
    init()
    parser = ArgumentParser()
    parser.add_argument("--num-gpus-per-node", 
                        type=int, 
                        required=True, 
                        help="specify number of GPUs/GCDs per node")
    parser.add_argument("--machine",
                        type=str,
                        required=True,
                        help="specify the machine you are running on. Will be used to create folders")
    parser.add_argument("--method",
                        type=str, 
                        choices=["mpi", "nccl", "mpi_mpi", "nccl_mpi", "mpi_nccl", "nccl_nccl", "mpird", "nccl_mpird"],
                        required=True)
    args = parser.parse_args()
    gpu_count, slurm_job_id = get_gpu_counts_and_job_id()
    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) 
    unit = "MB"

    is_hybrid = '_' in args.method
    use_rd = "mpird" in args.method     
    if is_hybrid:
        inner_pg, outer_pg = args.method.split("_")
        inner_pg_arg = inner_pg
        outer_pg_arg = "mpi" if outer_pg == "mpird" else outer_pg

        pg = ProcessGroups(args.num_gpus_per_node, 
                                   dist.get_world_size() // args.num_gpus_per_node, 
                                   inner_pg_arg,
                                   outer_pg_arg)
        args.method = f"inner_{inner_pg}_outer_{outer_pg}"
    elif args.method == "mpi" or args.method == "mpird":
        pg = MPI.COMM_WORLD
    else:
        pg = None
    
    data_folder = f"./data_10_runs/{args.machine}_3"
    os.makedirs(data_folder, exist_ok=True)

    csv_filename = os.path.join(data_folder,
                                f"gpus_{gpu_count}_slurm_{slurm_job_id}.csv")
    
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        header = ["gpu_count", "slurm_job_id", "output_size", "unit", f"time_{args.method}"]
        #header.extend([f"time_hybrid_{group_type}" for group_type in hybrid_pgs.keys()])
        writer.writerow(header)

        for size in sizes:
            if dist.get_rank() == 0:
                print(f"output size = {size} {unit}")
            mult = 2**20 if unit == "MB" else 2**10
            output_buffer_numel = size * (mult) // 2 
            input_buffer_numel = output_buffer_numel // dist.get_world_size()

            output_tensor = torch.empty((output_buffer_numel,), dtype=torch.bfloat16, device="cuda")
            input_tensor = torch.empty((input_buffer_numel,), dtype=torch.bfloat16, device="cuda")

            function = _all_gather if not is_hybrid else all_gather_2D
            time = time_something(function, output_tensor, input_tensor, group=pg, use_rd=use_rd)
           

            if dist.get_rank() == 0:
                print(f"time_{args.method} = {time:.2f} ms")                

            if dist.get_rank() == 0:
                print("===============================")
                writer.writerow([gpu_count, slurm_job_id, size, unit, time])
        
    dist.destroy_process_group()
