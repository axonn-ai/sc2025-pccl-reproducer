from mpi4py import MPI
import numpy as np
from utils import init
import torch

def ring_allgather(send_data, recv_data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Initialize recv_data with own data
    #recv_data[:] = 0
    chunk_size = send_data.numel()
    recv_data[rank*chunk_size:(rank+1)*chunk_size] = send_data  

    torch.cuda.current_stream().synchronize()
    # Create a window for one-sided communication on recv_data
    win = MPI.Win.Create(recv_data, disp_unit=1, comm=comm)
    
    for s in range(size - 1):
        src_rank = (rank - s) % size
        target_rank = (rank + 1) % size
        
        # Extract the data to send (as a contiguous array)
        #data_to_send = recv_data[src_rank:src_rank+1].copy()
        
        # Start the exposure epoch
        win.Fence(MPI.MODE_NOPRECEDE)
        
        # Put the data into the target's recv_data at the src_rank position
        win.Put(recv_data[src_rank*chunk_size:(src_rank+1)*chunk_size], target_rank, [src_rank*chunk_size*2, chunk_size*2])
        
        # End the exposure epoch
        win.Fence(MPI.MODE_NOSUCCEED)
    
    # Free the window
    win.Free()

def main():
    init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Each process contributes four integers integer
    send_data = np.array([rank, rank, rank, rank], dtype=np.int32)
    recv_data = np.zeros(size*send_data.size, dtype=np.int32)
    
    send_data = torch.from_numpy(send_data).to(torch.bfloat16).cuda()
    recv_data = torch.from_numpy(recv_data).to(torch.bfloat16).cuda()

    # Perform the ring all-gather
    ring_allgather(send_data, recv_data, comm)
    
    # Print the result
    print(f"Process {rank}: {send_data}")
    print(f"Process {rank}: {recv_data}")

if __name__ == "__main__":
    main()