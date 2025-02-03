import torch
import torch.distributed as dist
import numpy as np
import os
from mpi4py import MPI
from mpi_utils import _torch_to_mpi
import numpy as np
from tree_reduce_scatter_mpi import tree_reduce_scatter, pairwise_recursive_halving_reduce_scatter, ring_reduce_scatter

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
            # step 1: permute 
            output_msg_size = msg.size(0) // world_size
            input_splits = torch.split(msg, split_size_or_sections=output_msg_size)
            permuted_tensors = []
            for i in range(intra_node_group_size):
                idxes = list(np.arange(i, world_size, intra_node_group_size))
                permuted_tensors.extend([input_splits[idx] for idx in idxes])
            input_permuted = torch.cat(permuted_tensors)

            # step 2: intra-node reduce scatter using rccl
            output_intermediate = torch.empty((input_permuted.size(0) // intra_node_group_size,), 
                                 device=input_permuted.device, 
                                 dtype=input_permuted.dtype)
            dist.reduce_scatter_tensor(output_intermediate, input_permuted, group=intra_node_nccl_group)

            # step 3: inter-node all to all using MPI + device local reduce
            if False:
                output = torch.empty_like(output_intermediate)
                torch.cuda.current_stream().synchronize()
                inter_node_mpi_group.Alltoall(_torch_to_mpi(output_intermediate), _torch_to_mpi(output))
                ## device local reduce
                output = output.reshape(inter_node_group_size, -1).sum(dim=0)
            else:
                output = torch.empty((output_intermediate.size(0) // inter_node_group_size,), 
                                      device=output_intermediate.device,
                                      dtype=output_intermediate.dtype)
                #tree_reduce_scatter(output_intermediate, output, comm=inter_node_mpi_group)
                pairwise_recursive_halving_reduce_scatter(output_intermediate, output, comm=inter_node_mpi_group)
                #ring_reduce_scatter(output_intermediate, output, comm=inter_node_mpi_group)

        elif method == "nccl":
            output = torch.empty(msg.size(0) // world_size, device=msg.device, dtype=msg.dtype)
            dist.reduce_scatter_tensor(output, msg)
        elif method == "mpi":
            output = torch.empty(msg.size(0) // world_size, device=msg.device, dtype=msg.dtype)
            pairwise_recursive_halving_reduce_scatter(msg, output)

        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
        # print(times[-1] / 2)

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
        hybrid = time_reduce_scatter(SZ, method="hybrid", intra_node_gpu_count=8)
        if dist.get_rank() == 0:
            print("===============================")
        bws["rccl"].append(rccl)
        bws["hybrid"].append(hybrid)
        bws["mpi"].append(mpi)
    if dist.get_rank() == 0:
        print(bws)
    #print(output)
    dist.destroy_process_group()
