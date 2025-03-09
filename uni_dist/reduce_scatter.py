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

def ring_reduce_scatter_mpi(output_tensor: torch.Tensor,
                        input_tensor: torch.Tensor,
                        group: Optional[MPI.Comm] = None,
                        async_op: bool = False, 
                        op=torch.add):
    """
    Performs a ring-based reduce-scatter using torch tensors.
    
    Each process holds a 2D tensor 'sendbuf' of shape (P, N), where P is the number 
    of processes and N is the block size. Each row of the tensor corresponds to a block.
    The goal is to reduce the i-th block (row) from all processes (using the provided op)
    so that at the end, each process gets the fully reduced block corresponding to its 
    assigned segment.
    
    The algorithm performs P-1 steps. At each step, each process:
      - Sends a designated block to its right neighbor.
      - Receives a block from its left neighbor.
      - Immediately reduces the received block into its local copy.
      
    After the loop, the fully reduced block is at index (rank - (P-1)) % P.
    """
    assert not async_op, "non-blocking primitives not supported"
    comm = MPI.COMM_WORLD if group is None else group
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    assert input_tensor.dim() == 1, "input tensor should be 1D"
    assert output_tensor.dim() == 1, "output tensor should be 1D"

    sendbuf = input_tensor.view(size, -1)

    # Ensure that there is one block per process.
    assert sendbuf.size(0) == size, "sendbuf must have one block per process"
    block_size = sendbuf.size(1)
    
    # Work on a local copy of the buffer.
    buf = sendbuf.clone()
    tmp = torch.empty(block_size, dtype=sendbuf.dtype, device=sendbuf.device)
    
    # Perform size-1 steps of the ring reduce-scatter.
    for step in range(size - 1):
        # Determine the block indices.
        send_idx = (rank - step) % size
        recv_idx = (rank - step - 1) % size
        
        # Identify neighbors in the ring.
        dest = (rank + 1) % size
        source = (rank - 1 + size) % size
        
        # Copy the block to be sent.
        send_data = buf[send_idx] #.clone()
        
        # Send the designated block to the right and receive a block from the left.
        torch.cuda.current_stream().synchronize()
        comm.Sendrecv(sendbuf=send_data, dest=dest, sendtag=0,
                      recvbuf=tmp, source=source, recvtag=0)
        
        # Reduce the received block into our local block.
        buf[recv_idx] = op(buf[recv_idx], tmp)
    
    # The final, fully reduced block is at this index.
    final_idx = (rank - (size - 1)) % size
    output_tensor.copy_(buf[final_idx])


def _reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    directly_call_mpi: bool = False
) -> Optional[Request]:

    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):
        # Delegate to torch.distributed.all_gather
        request = dist.reduce_scatter_tensor(output_tensor, 
                                             input_tensor, 
                                             group=group, 
                                             async_op=async_op)
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        # make sure that the cpu is synchronized with the current stream
        if not directly_call_mpi:
            request = ring_reduce_scatter_mpi(output_tensor, input_tensor, group, async_op)
        else:
            request = group.Reduce_scatter_block(input_tensor, output_tensor)
    else:
        raise TypeError(
            f"Unsupported group type: {type(group)}. "
            "Expected torch.distributed.ProcessGroup or mpi4py.MPI.Comm."
        )
    return request 

def reduce_scatter_2D(output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[ProcessGroups] = None,
    async_op: bool = False):

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
    _reduce_scatter(output_intermediate, input_permuted, group.get_inner_group(), async_op=False)

    # Step-2 inter-node 
    _reduce_scatter(output_tensor, output_intermediate, group.get_outer_group(), async_op=False)


