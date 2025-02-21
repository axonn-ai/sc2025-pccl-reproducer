from typing import Optional, Tuple
import torch
from mpi4py import MPI
import torch.distributed as dist

def _all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Tuple[dist.ProcessGroup, MPI.Comm]] = None,
):
    """
    Hierarchical all-gather with interleaved phases.

    Phase 1 (Inter-node):  
      Each process (grouped in inter_comm, which contains GPUs from different nodes that share
      the same local rank) runs an MPI ring all-gather. The gathered data is stored in
      inter_buffer (shape [num_nodes, tensor_size]), where tensor_size = input_tensor.numel().

    Phase 2 (Intra-node, interleaved):  
      As soon as each row of inter_buffer becomes available, an asynchronous intra-node all_gather
      is launched over intra_comm (a NCCL-based torch.distributed ProcessGroup) to gather that row
      from every GPU on the node. This yields, for each inter_buffer row, a tensor of shape
      [intra_size, tensor_size]. Finally, the per-row gathered results are rearranged to produce
      a global result of shape [intra_size * num_nodes, tensor_size] which is copied into output_tensor.

    Args:
      output_tensor (torch.Tensor): Preallocated tensor to hold the final gathered data.
          It should be viewable as a 2D tensor of shape [global_size, -1] with global_size = intra_size * num_nodes.
      input_tensor (torch.Tensor): The local tensor to be gathered.
      group (Optional[Tuple[dist.ProcessGroup, MPI.Comm]]): Tuple (intra_comm, inter_comm) where:
            - intra_comm is a NCCL-based torch.distributed ProcessGroup for intra-node communication.
            - inter_comm is an MPI communicator for inter-node communication.
          Must not be None.
    """
    if group is None:
        raise ValueError("group must be provided as a tuple: (intra_comm, inter_comm)")
    
    # Unpack the two groups.
    intra_comm, inter_comm = group.get_inner_group(), group.get_outer_group()

    # --- Inter-node Phase (MPI Ring All-gather) ---
    inter_rank = inter_comm.Get_rank()
    num_nodes = inter_comm.Get_size()

    # Flatten the input tensor.
    local_flat = input_tensor.view(-1)
    tensor_size = local_flat.numel()

    # Allocate an inter_buffer to hold one row per node.
    # Shape: [num_nodes, tensor_size]
    inter_buffer = torch.empty((num_nodes, tensor_size), dtype=input_tensor.dtype, device=input_tensor.device)
    # Our own data goes into the slot corresponding to our inter_comm rank.
    inter_buffer[inter_rank].copy_(local_flat)

    # We'll launch an asynchronous intra-node all_gather on each row as soon as it is ready.
    # Store the async work handles and the gathered lists in a dictionary keyed by row index.
    intra_async_results = {}
    # Get the intra-node group size.
    intra_size = dist.get_world_size(group=intra_comm)

    # Launch intra-node all_gather for our own row (which is immediately available).
    gather_list = [torch.empty((tensor_size,), dtype=input_tensor.dtype, device=input_tensor.device) for _ in range(intra_size)]
    work = dist.all_gather(gather_list, inter_buffer[inter_rank], group=intra_comm, async_op=True)
    intra_async_results[inter_rank] = (work, gather_list)

    # Run the MPI ring loop for num_nodes - 1 iterations.
    current_data = local_flat.clone()
    for i in range(num_nodes - 1):
        recv_tensor = torch.empty_like(local_flat)
        if input_tensor.is_cuda:
            torch.cuda.current_stream().synchronize()
        dest = (inter_rank + 1) % num_nodes
        source = (inter_rank - 1 + num_nodes) % num_nodes
        req_send = inter_comm.Isend(current_data, dest=dest, tag=0)
        req_recv = inter_comm.Irecv(recv_tensor, source=source, tag=0)
        MPI.Request.Waitall([req_send, req_recv])
        # Determine the slot (row index) where the received data belongs.
        slot = (inter_rank - i - 1 + num_nodes) % num_nodes
        inter_buffer[slot].copy_(recv_tensor)
        current_data = recv_tensor.clone()

        # Launch an asynchronous intra-node all_gather for this newly received row.
        gather_list = [torch.empty((tensor_size,), dtype=input_tensor.dtype, device=input_tensor.device) for _ in range(intra_size)]
        work = dist.all_gather(gather_list, inter_buffer[slot], group=intra_comm, async_op=True)
        intra_async_results[slot] = (work, gather_list)

    # --- Wait for all intra-node asynchronous all_gather operations to complete ---
    for slot, (work, _) in intra_async_results.items():
        work.wait()

    # --- Final Assembly ---
    # For each inter_buffer row (indexed 0 .. num_nodes-1), retrieve the gathered intra-node data.
    # Each gathered result is a list of 'intra_size' tensors (each of shape [tensor_size]).
    # We stack each such list to form a tensor of shape [intra_size, tensor_size].
    # Then, we want the final global result to have shape [intra_size * num_nodes, tensor_size].
    gathered_rows = []
    for r in range(num_nodes):
        _, gather_list = intra_async_results[r]
        row_tensor = torch.stack(gather_list, dim=0)  # shape: [intra_size, tensor_size]
        gathered_rows.append(row_tensor)
    # Now, gathered_rows is a list of num_nodes tensors, each of shape [intra_size, tensor_size].
    # We first stack them along a new dimension to form a tensor of shape [num_nodes, intra_size, tensor_size],
    # then transpose to [intra_size, num_nodes, tensor_size] and finally reshape.
    stacked = torch.stack(gathered_rows, dim=0)         # shape: [num_nodes, intra_size, tensor_size]
    transposed = stacked.transpose(0, 1)                # shape: [intra_size, num_nodes, tensor_size]
    global_result = transposed.reshape(intra_size * num_nodes, tensor_size)
    
    # Copy the final global result into output_tensor.
    output_tensor.copy_(global_result.view(-1))
