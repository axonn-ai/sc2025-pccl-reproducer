# write an all gather function that can use either nccl or mpi.
#  
# all_gather.py

import torch
import torch.distributed as dist
from mpi4py import MPI
from typing import List, Optional, Union
from .request import Request
from .process_groups import ProcessGroups
import numpy as np
from .utils import _torch_to_mpi
from .all_gather import _all_gather


import torch
import math
from mpi4py import MPI
from typing import Optional

def _all_reduce(
    input_tensor: torch.Tensor, # all-reduce is inplace.
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    use_rh_and_rd: bool = False,
    use_yacl: bool = False,
    directly_call_mpi = False,
) -> Optional[Request]:

    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):
        # Delegate to torch.distributed.all_gather
        request = dist.all_reduce(input_tensor, 
                                group=group, 
                                async_op=async_op)
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        # make sure that the cpu is synchronized with the current stream
        if not directly_call_mpi:
            if use_yacl:
                import yacl 
                yacl.all_reduce_mpi(input_tensor, group, "recursive" if use_rh_and_rd else "ring")
            
        else:
            output_tensor = torch.empty_like(input_tensor)
            request = group.Allreduce(input_tensor, output_tensor)
            input_tensor.copy_(output_tensor)

        request = None
    else:
        raise TypeError(
            f"Unsupported group type: {type(group)}. "
            "Expected torch.distributed.ProcessGroup or mpi4py.MPI.Comm."
        )
    return request 

def reduce_scatter_2D(output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[ProcessGroups] = None,
    async_op: bool = False,
    use_rh: bool = False):

    assert not async_op, "Non blocking version not implemented"

    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_gather_2D only admits 1D tensors"
    intra_node_group_size, inter_node_group_size = group.get_world_size()

    # step 1: on-device permutation 
    world_size = inter_node_group_size * intra_node_group_size
    output_msg_size = input_tensor.size(0) // world_size
    input_splits = torch.split(input_tensor, split_size_or_sections=output_msg_size)
    permuted_tensors = []
    for i in range(intra_node_group_size):
        idxes = list(np.arange(i, world_size, intra_node_group_size))
        permuted_tensors.extend([input_splits[idx] for idx in idxes])
    input_permuted = torch.cat(permuted_tensors)

    # Step-2 intra-node reduce-scatter
    output_intermediate = torch.empty(input_permuted.size(0) // intra_node_group_size, 
                                      device=input_tensor.device, 
                                      dtype=input_tensor.dtype)
    _reduce_scatter(output_intermediate, input_permuted, group.get_inner_group(), async_op=False, use_rh=use_rh)

    # Step-2 inter-node 
    _reduce_scatter(output_tensor, output_intermediate, group.get_outer_group(), async_op=False, use_rh=use_rh)





