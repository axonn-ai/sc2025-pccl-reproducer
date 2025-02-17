import torch
import torch.distributed as dist
import numpy as np
import os
from mpi4py import MPI
from uni_dist.process_groups import _torch_to_mpi
import numpy as np
from reduce_scatter import reduce_scatter as hybrid_reduce_scatter
from reduce_scatter import pairwise_recursive_halving_reduce_scatter
from uni_dist.process_groups import create_2D_grid

@torch.no_grad()
def time_reduce_scatter(SZ, method="nccl", **kwargs):
    world_rank = dist.get_rank() 
    world_size = dist.get_world_size()
    assert SZ % world_size == 0
    gpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(world_rank%gpus_per_node)
    msg = torch.full((SZ,), 5,  dtype=torch.bfloat16, device="cuda")
    st, en = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    if method == "hybrid":
        intra_node_gpu_count = kwargs.get("intra_node_gpu_count", torch.cuda.device_count())
        hybrid_process_groups = create_2D_grid(intra_node_gpu_count, dist.get_world_size() // intra_node_gpu_count)

    for _ in range(20):
        st.record()
        if method == "hybrid":
            output = hybrid_reduce_scatter(msg, hybrid_process_groups) 
        elif method == "nccl":
            output = torch.empty(msg.size(0) // world_size, device=msg.device, dtype=msg.dtype)
            dist.reduce_scatter_tensor(output, msg)
        elif method == "mpi":
            output = torch.empty(msg.size(0) // world_size, device=msg.device, dtype=msg.dtype)
            pairwise_recursive_halving_reduce_scatter(msg, output)

        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
        

    input_size = SZ * 2  # bytes
    g = world_size
    bw = (g - 1) / g * input_size / 1e9 / np.mean(times[-10:]) * 1000

    if world_rank == 0:
        print(f"Method = {method}")
        print(
            f"Reduce scatter bus bw for {g} GPUs is {bw:.3f} GBPS for message "
            f"output size {output.size(0)* 2 / 1e6:.3f} MB"
        )
        print(f"time = {np.mean(times[-10:]):.3f} ms")
        #print(output[:3])
    return bw


if __name__ == "__main__":
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    dist.init_process_group(rank=rank, 
                            world_size=world_size,
                            backend="nccl", 
                            init_method="env://")
    bs=1
    sq=2048
    hdim=4096
    sizes_mb = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

    bws = {"rccl": [], "hybrid": [], "mpi": []}

    for size_mb in sizes_mb:
        if dist.get_rank() == 0:
            print(f"input size = {size_mb} MB")
        SZ = size_mb * (2**20) // 2  #// dist.get_world_size()
        mpi = time_reduce_scatter(SZ, method="mpi")
        rccl = time_reduce_scatter(SZ, method="nccl")
        hybrid = time_reduce_scatter(SZ, method="hybrid", intra_node_gpu_count=4)
        if dist.get_rank() == 0:
            print("===============================")
        bws["rccl"].append(rccl)
        bws["hybrid"].append(hybrid)
        bws["mpi"].append(mpi)
    if dist.get_rank() == 0:
        print(bws)
    #print(output)
    dist.destroy_process_group()
