from uni_dist.process_groups import create_2D_grid, _torch_to_mpi
import torch 
import torch.distributed as dist
import numpy as np


@torch.no_grad()
def all_gather(msg, process_groups, lib="hybrid-nccl-mpi"):
    assert lib in ["hybrid-nccl-mpi", "hybrid-nccl-nccl", "nccl"]
    if lib == "nccl":
        assert len(process_groups) == 1
        world_size = dist.get_world_size(process_groups[0])
        output = [torch.empty_like(msg) for _ in range(world_size)]
        output = torch.cat(output)          
        dist.all_gather_into_tensor(output, msg)
        return output

    intra_node_group, inter_node_group = process_groups
    if lib == "hybrid-nccl-mpi":
        inter_node_group_size = inter_node_group.Get_size()
    elif lib == "hybrid-nccl-nccl":
        inter_node_group_size = dist.get_world_size(inter_node_group)

    intra_node_group_size = dist.get_world_size(intra_node_group)

    # inter-node all-gather
    output_inter_node = torch.empty(msg.size(0) * inter_node_group_size, device=msg.device, dtype=msg.dtype)
    if lib == "hybrid-nccl-mpi":
        output_inter_node_mpi = _torch_to_mpi(output_inter_node)
        torch.cuda.current_stream().synchronize()
        inter_node_group.Allgather(msg, output_inter_node_mpi)
    else:
        dist.all_gather_into_tensor(output_inter_node, msg, group=inter_node_group) 

    # intra-node all-gather
    output = torch.empty(output_inter_node.size(0) * intra_node_group_size, device=msg.device, dtype=msg.dtype)
    dist.all_gather_into_tensor(output, output_inter_node, group=intra_node_group)
    output_splits = torch.split(output, split_size_or_sections=msg.size(0))
    
    # inter_node_group_size * intra_node_group_size
    ordered_tensors = []
    for i in range(inter_node_group_size):
        idxes = list(np.arange(i, intra_node_group_size*inter_node_group_size, inter_node_group_size ))
        ordered_tensors.extend([output_splits[idx] for idx in idxes])
    output = torch.cat(ordered_tensors)
    return output

    