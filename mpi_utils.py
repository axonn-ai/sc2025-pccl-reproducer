import torch
from mpi4py import MPI
import numpy as np
import torch.distributed as dist

def _torch_to_mpi(tensor: torch.Tensor):
    """Converts a PyTorch tensor into an mpi4py compatible array using its
    unified virtual address

    Arguments:
        tensor (torch.Tensor): the Pytorch tensor
    """
    return [
        MPI.memory.fromaddress(
            tensor.data_ptr(), tensor.element_size() * tensor.nelement()
        ),
        MPI.FLOAT,
    ]

def create_2D_grid(intra_group_size, inter_group_size, outer_group="mpi"):
    # sometimes torch cuda device count is not reliable. For instance one might only 
    # expose 1 GPU to each process in the srun command. 
    assert outer_group in ["mpi", "nccl"]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size % (intra_group_size * inter_group_size) == 0
    num_2d_grids = world_size // (intra_group_size * inter_group_size)
    process_group_grid = np.arange(world_size).reshape(num_2d_grids, inter_group_size, intra_group_size)
    
    # create intra-node NCCL process groups
    for i in range(num_2d_grids):
        for j in range(inter_group_size):
            ranks = list(process_group_grid[i, j])
            this_intra_node_group = dist.new_group(ranks=ranks, backend="nccl") 
            if rank in ranks:
                intra_node_nccl_group = this_intra_node_group

    if outer_group == "nccl":
        # create intra-node NCCL process groups
        for i in range(num_2d_grids):
            for j in range(inter_group_size):
                ranks = list(process_group_grid[i, j])
                this_intra_node_group = dist.new_group(ranks=ranks, backend="nccl") 
                if rank in ranks:
                    intra_node_nccl_group = this_intra_node_group
            
            for j in range(intra_group_size):
                ranks = list(process_group_grid[i, :, j])
                this_inter_node_group = dist.new_group(ranks=ranks, backend="nccl") 
                if rank in ranks:
                    inter_node_nccl_group = this_inter_node_group
        groups = [intra_node_nccl_group, inter_node_nccl_group]


    elif outer_group == "mpi":
        # create inter-node MPI process_groups 
        # unique color based on the first and third dimension 
        color = rank % intra_group_size + (rank // (intra_group_size * inter_group_size)) * (intra_group_size) 
        inter_node_mpi_group = MPI.COMM_WORLD.Split(color)
        groups = [intra_node_nccl_group, inter_node_mpi_group]
    return groups

