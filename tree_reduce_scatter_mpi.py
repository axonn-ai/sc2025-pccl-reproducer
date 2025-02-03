from mpi4py import MPI
import torch
import math

@torch.no_grad()
def tree_reduce_scatter(sendbuf, recvbuf, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert sendbuf.dim() == 1 and recvbuf.dim() == 1
    chunk_size = recvbuf.size(0)

    parent = (rank - 1) // 2 if rank != 0 else -1  # Root has no parent
    left_child = 2 * rank + 1
    right_child = 2 * rank + 2

    # Ensure the data is evenly divided
    assert sendbuf.size(0) == chunk_size * size, "Data must be evenly distributed across processes"

    # Reduction Phase: Receive from children and reduce
    for tag, child in enumerate([left_child, right_child]):
        if child < size:
            child_data = torch.empty_like(sendbuf)
            torch.cuda.current_stream().synchronize()
            comm.Recv(child_data, source=child, tag=tag)
            sendbuf.add_(child_data) 

    # Send to parent if not root
    if rank != 0:
        torch.cuda.current_stream().synchronize()
        comm.Send(sendbuf, dest=parent, tag=rank - (2*parent+1))

    comm.Scatter(sendbuf, recvbuf)
    


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
    max_steps = int(math.log2(size))

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




def ring_reduce_scatter(sendbuf, recvbuf, comm=MPI.COMM_WORLD):
    """
    Ring-based reduce-scatter algorithm.

    We conceptually break the global array (size = chunk_size*P) into P blocks,
    each block of length chunk_size. Rank r is responsible for block r.
    By doing (P-1) passes (shifting blocks in a ring), each block accumulates
    the sum of all ranks' data.

    Complexity Analysis:
      - Communication: Each of the P ranks sends and receives chunk_size data in each of (P-1) steps.
        => O(P * chunk_size * P) = O(P^2 * chunk_size) = O(N*P), where N=chunk_size*P.
      - Latency: P-1 iterations.
      - Good for large messages, but has linear complexity in P.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert sendbuf.dim() == 1 and recvbuf.dim() == 1, "Tensors must be 1D"
    chunk_size = recvbuf.size(0)
    assert sendbuf.size(0) == chunk_size * size, "Data must be evenly distributed"

    left = (rank - 1) % size
    right = (rank + 1) % size

    # We'll store each of the P blocks in an array of PyTorch slices.
    # For convenience, let's do it in one big tensor, but we keep track of each block’s offset.
    blocks = []
    for block_idx in range(size):
        start = block_idx * chunk_size
        end   = (block_idx + 1) * chunk_size
        blocks.append(sendbuf[start:end])  # each block is a view

    # We will do (size - 1) rounds.
    # On round s, we send block (rank - s) mod size to the left,
    # and receive block (rank - s + 1) mod size from the right, summation in place.
    temp = torch.empty(chunk_size, dtype=sendbuf.dtype, device=sendbuf.device)

    for s in range(size - 1):
        # Which block am I sending this round?
        send_block_idx = (rank - s) % size
        recv_block_idx = (rank - s + 1) % size

        # 1) Send my "send_block_idx" block to the left neighbor
        torch.cuda.current_stream().synchronize()
        req_send = comm.Isend(blocks[send_block_idx], dest=left, tag=s)

        # 2) Receive into 'temp' from my right neighbor
        torch.cuda.current_stream().synchronize()
        req_recv = comm.Irecv(temp, source=right, tag=s)

        req_send.Wait()
        req_recv.Wait()

        # 3) Sum it into my local copy of the corresponding block
        blocks[recv_block_idx].add_(temp)

    # After P-1 shifts, block r holds the sum of that block across all ranks.
    # Copy block[rank] into recvbuf
    recvbuf.copy_(blocks[rank])


