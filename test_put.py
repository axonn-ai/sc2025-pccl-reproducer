import torch
from mpi4py import MPI
from utils import init
import torch.distributed as dist

def ring_reduce_scatter(sendbuf, op=torch.add):
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
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
    return buf[final_idx]

if __name__ == '__main__':
    init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    block_size = 4  # for example, each block is a vector of 4 elements
    # Each process creates a tensor of shape (size, block_size) filled with its rank+1.
    data = torch.full((size, block_size), fill_value=rank + 1, dtype=torch.bfloat16, device="cuda")
    
    # Perform the reduce-scatter using torch tensors.
    result = ring_reduce_scatter(data, op=torch.add)
    
    # With 8 ranks and addition, each reduced block should contain the sum 1+2+...+8 = 36.
    print(f"Rank {rank} got result: {result}")

    dist.destroy_process_group()
