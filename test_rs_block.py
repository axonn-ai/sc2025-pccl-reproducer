from mpi4py import MPI
import torch
from utils import init
from uni_dist.utils import _torch_to_mpi
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from torch.profiler import _KinetoProfile
import torch.distributed as dist
from utils import time_something
_KinetoProfile._get_distributed_info = lambda self: None

init()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define tensor sizes
bytes_per_elem = 4
buffer_size_mb = 512
full_size = buffer_size_mb * 2**20 // bytes_per_elem
chunk_size = full_size // size

dtype = torch.float32
# Create a bfloat16 tensor with values based on rank
tensor = torch.full((full_size,), rank, dtype=dtype)

# Output buffer for receiving the reduced chunk
recv_tensor = torch.empty((chunk_size,), dtype=dtype)

# Define a custom MPI operation for bfloat16 addition
def bf16_sum(a, b, dtype):
    """ Custom reduction function for bfloat16 tensors """
    a_tensor = torch.frombuffer(a, dtype=torch.bfloat16).cuda()
    b_tensor = torch.frombuffer(b, dtype=torch.bfloat16)
    b_tensor_cuda = b_tensor.cuda()
    b_tensor.copy_(a_tensor + b_tensor_cuda)  # Perform element-wise addition in-place

# Create the custom MPI operation (non-commutative flag is False)
bf16_op = MPI.Op.Create(bf16_sum, commute=True)

# Perform Reduce_scatter_block using the custom bfloat16 sum operation


rs_block_time = time_something(comm.Reduce_scatter_block, sendbuf=_torch_to_mpi(tensor), recvbuf=_torch_to_mpi(recv_tensor),)
bus_bw = (size-1)/size * buffer_size_mb / rs_block_time

print(f"Time = {rs_block_time:.2f} ms | B/W = {bus_bw:.2f} GB/s")

# Free the custom MPI operation after use
bf16_op.Free()
dist.destroy_process_group()
