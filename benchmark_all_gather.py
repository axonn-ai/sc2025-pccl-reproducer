import torch
import torch.distributed as dist
import numpy as np
import os
from mpi4py import MPI
from mpi_utils import _torch_to_mpi
import numpy as np

@torch.no_grad()
def time_all_gather(SZ, method="nccl", **kwargs):
    world_rank = dist.get_rank() 
    gpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(world_rank%gpus_per_node)
    msg = torch.full((SZ,), world_rank,  dtype=torch.bfloat16, device="cuda")
    st, en = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    if method == "hybrid":
        # sometimes torch cuda device count is not reliable. For instance one might only 
        # expose 1 GPU to each process in the srun command. 
        intra_node_gpu_count = kwargs.get("intra_node_gpu_count", torch.cuda.device_count())
        rank = dist.get_rank()
        world_size = dist.get_world_size() 
        assert world_size % intra_node_gpu_count == 0 
        process_group_grid = np.arange(world_size).reshape(-1, intra_node_gpu_count)
        # print(process_group_grid)
        # exit()
        intra_node_group_size = process_group_grid.shape[1]
        inter_node_group_size = process_group_grid.shape[0]

        # create intra-node NCCL process groups
        for i in range(inter_node_group_size):
            ranks = list(process_group_grid[i])
            this_intra_node_group = dist.new_group(ranks=ranks, backend="nccl") 
            if rank in ranks:
                intra_node_nccl_group = this_intra_node_group

        # create inter-node MPI process_groups 
        inter_node_mpi_group = MPI.COMM_WORLD.Split(rank % intra_node_gpu_count)

    for _ in range(20):
        st.record()
        if method == "hybrid":
            ## first inter-node all-gather
            output_inter_node = torch.empty(msg.size(0) * inter_node_group_size, device=msg.device, dtype=msg.dtype)
            output_inter_node_mpi = _torch_to_mpi(output_inter_node)
            torch.cuda.current_stream().synchronize()
            inter_node_mpi_group.Allgather(msg, output_inter_node_mpi)
            # output = [torch.empty_like(output_inter_node) for _ in range(intra_node_group_size)]
            # output = torch.cat(output)
            output = torch.empty(output_inter_node.size(0) * intra_node_group_size, device=msg.device, dtype=msg.dtype)
            dist.all_gather_into_tensor(output, output_inter_node, group=intra_node_nccl_group)
            output_splits = torch.split(output, split_size_or_sections=msg.size(0))
            # inter_node_group_size * intra_node_group_size
            ordered_tensors = []
            for i in range(inter_node_group_size):
                idxes = list(np.arange(i, world_size, inter_node_group_size ))
                ordered_tensors.extend([output_splits[idx] for idx in idxes])
            output = torch.cat(ordered_tensors)
        else:
            output = [torch.empty_like(msg) for _ in range(dist.get_world_size())]
            output = torch.cat(output)
            if method == "nccl":            
                dist.all_gather_into_tensor(output, msg)
            elif method == "mpi":
                output_mpi = _torch_to_mpi(output)
                MPI.COMM_WORLD.Allgather(msg, output_mpi)
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
        # print(times[-1] / 2)

    input_size = SZ * 2  # bytes
    g = torch.distributed.get_world_size()
    bw = (g - 1) * input_size / 1e9 / np.mean(times[-10:]) * 1000

    if world_rank == 0:
        print(f"Method = {method}")
        print(
            f"All-gather bus bw for {g} GPUs is {bw:.3f} GBPS for message "
            f"output size {output.size(0)* 2 / 1e6:.3f} MB"
        )
        print(f"time = {np.mean(times[-10:]):.3f} ms")
    
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
    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) 
    unit = "MB"
    

    bws = {"mpi": [], "rccl": [], "hybrid": []}

    for size in sizes:
        if dist.get_rank() == 0:
            print(f"output size = {size} {unit}")
        mult = 2**20 if unit == "MB" else 2**10
        SZ = size * (mult) // 2 // dist.get_world_size()
        rccl = time_all_gather(SZ, method="nccl")
        mpi = time_all_gather(SZ, method="mpi")
        hybrid = time_all_gather(SZ, method="hybrid", intra_node_gpu_count=8)
        if dist.get_rank() == 0:
            print("===============================")
        bws["mpi"].append(mpi)
        bws["rccl"].append(rccl)
        bws["hybrid"].append(hybrid)
    if dist.get_rank() == 0:
        print(bws)
    #print(output)
    dist.destroy_process_group()
