import torch 
from mpi4py import MPI
import torch.distributed as dist
import numpy as np

@torch.no_grad()
def reduce_scatter(msg, process_groups, comm_lib="hybrid"):
    if comm_lib == "nccl":
        assert len(process_groups) == 1
        world_size = dist.get_world_size(process_groups[0])
        output = torch.empty(msg.size(0) // world_size, device=msg.device, dtype=msg.dtype)
        dist.reduce_scatter_tensor(output, msg)
        return output

    intra_node_nccl_group, inter_node_mpi_group = process_groups
    inter_node_group_size = inter_node_mpi_group.Get_size()
    intra_node_group_size = dist.get_world_size(intra_node_nccl_group)

    # step 1: permute 
    world_size = inter_node_group_size * intra_node_group_size
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
    # if False:
    #     output = torch.empty_like(output_intermediate)
    #     torch.cuda.current_stream().synchronize()
    #     inter_node_mpi_group.Alltoall(_torch_to_mpi(output_intermediate), _torch_to_mpi(output))
    #     ## device local reduce
    #     output = output.reshape(inter_node_group_size, -1).sum(dim=0)
    # else:
    output = torch.empty((output_intermediate.size(0) // inter_node_group_size,), 
                            device=output_intermediate.device,
                            dtype=output_intermediate.dtype)
    #tree_reduce_scatter(output_intermediate, output, comm=inter_node_mpi_group)
    pairwise_recursive_halving_reduce_scatter(output_intermediate, output, comm=inter_node_mpi_group)
    return output

@torch.no_grad()
def pairwise_recursive_halving_reduce_scatter(sendbuf, recvbuf, comm=MPI.COMM_WORLD):
    """
    Pairwise exchange (recursive halving) reduce-scatter.

    Each rank has the entire dataset in sendbuf (length = chunk_size*world_size),
    and ends with just its own sub-chunk (length = chunk_size) in recvbuf.

    Complexity Analysis:
      - Communication: O(log P) steps.
      - Each step roughly sends/receives half of the "active" data. 
        Worst case data volume ~ N * log P, where N = chunk_size * P and P = size.
      - More scalable than gather-scatter for large P.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Basic checks
    assert sendbuf.dim() == 1 and recvbuf.dim() == 1, "Tensors must be 1D"
    chunk_size = recvbuf.size(0)
    assert sendbuf.size(0) == chunk_size * size, "Data must be evenly distributed"

    # We'll do the exchange in-place in a working buffer.
    # Start with a clone of 'sendbuf' so we don't overwrite original.
    work_buf = torch.clone(sendbuf)

    # The idea: after k steps, each rank "owns" certain contiguous segments
    # in 'work_buf'. We identify which segments belong to our partner,
    # exchange them, sum them in place, etc.

    # Track the "range" of indices that rank currently owns
    # (in a typical recursive-halving approach, the owned segments shrink each step).
    # But for simplicity, we can do a standard approach where in step k,
    # we exchange half the ranks that differ by 2^k in the upper or lower half.
    # We'll skip the complex indexing details and assume the entire buffer is always in play
    # but we only add the portion relevant to us.

    step = 0

    # Current segment size that we keep
    seg_size = chunk_size * size  # starts as the full buffer
    # We'll reduce seg_size by half each iteration.

    mask = 1
    while mask < size:
        partner = rank ^ mask

        # Determine which half belongs to 'partner' vs 'me'.
        # For step k, the block size = seg_size/2
        half_size = seg_size // 2

        if (rank & mask) == 0:
            # I keep the lower half, partner keeps the upper half
            send_slice = work_buf[half_size:seg_size]
            # I'll receive the upper half from partner
            recv_slice = torch.empty_like(send_slice)
        else:
            # I keep the upper half, partner keeps the lower half
            send_slice = work_buf[0:half_size]
            recv_slice = torch.empty_like(send_slice)

        # Exchange data with partner
        torch.cuda.current_stream().synchronize()
        # req_recv = comm.Irecv(recv_slice, source=partner, tag=step)
        # req_send = comm.Isend(send_slice, dest=partner, tag=step)
        # req_send.Wait()
        # req_recv.Wait()
        comm.Sendrecv(
            sendbuf=send_slice, dest=partner, sendtag=step,
            recvbuf=recv_slice, source=partner, recvtag=step
        )

        # Locally sum the received portion
        # (this is the "reduce" part)
        send_slice.add_(recv_slice)

        # If I'm keeping the *lower* half, the "active" portion is in [0 : half_size]
        # If I'm keeping the *upper* half, it's in [half_size : seg_size].
        if (rank & mask) == 0:
            # Lower half is now in [0 : half_size] after summation
            # so we move it into that region:
            #work_buf[0:half_size].copy_(work_buf[half_size:seg_size])
            work_buf = work_buf[half_size:]
        # else we keep the upper half as is in [0 : half_size], so no copy needed

        # The "active" segment is now half_size
        seg_size = half_size
        mask <<= 1
        step += 1

    # By now, seg_size == chunk_size, and the rank's final portion is in work_buf[0:chunk_size].
    recvbuf.copy_(work_buf[0:chunk_size])

