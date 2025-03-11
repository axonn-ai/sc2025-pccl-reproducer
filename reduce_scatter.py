import torch 
from mpi4py import MPI
import torch.distributed as dist
import numpy as np
from uni_dist import _reduce_scatter
from utils import init 

if __name__ == '__main__':
    init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # This test requires CUDA.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, this test requires a CUDA device.")

    # Optionally, set the CUDA device based on rank (assuming one GPU per MPI process).
    #torch.cuda.set_device(rank)

    block_size = 4  # Each block will have 4 elements.
    # Create a local input tensor of shape (P * block_size,).
    # Each process fills its tensor with P blocks such that the j-th block contains (rank*10 + j)
    blocks = []
    input_tensor_rh = torch.randn((block_size * size,), device="cuda", dtype=torch.float32)
    input_tensor_nccl = input_tensor_rh.detach().clone()
    input_tensor_ring = input_tensor_rh.detach().clone()

 
    # Pre-allocate an output tensor to hold the reduced result (one block).
    output_tensor_rh = torch.empty((block_size,), dtype=torch.float32, device="cuda")
    output_tensor_nccl = torch.empty((block_size,), dtype=torch.float32, device="cuda")
    output_tensor_ring = torch.empty((block_size,), dtype=torch.float32, device="cuda")

    _reduce_scatter(output_tensor_rh, input_tensor_rh, MPI.COMM_WORLD, use_rh=True, directly_call_mpi=False)
    _reduce_scatter(output_tensor_nccl, input_tensor_nccl, None)
    _reduce_scatter(output_tensor_ring, input_tensor_ring, MPI.COMM_WORLD, use_rh=False, directly_call_mpi=False)

    print((output_tensor_nccl-output_tensor_rh).abs().mean(), 
          (output_tensor_nccl-output_tensor_ring).abs().mean())

    torch.cuda.synchronize()

    