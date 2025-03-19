from deepspeed.comm.torch import TorchBackend
from deepspeed.comm.reduce_op import ReduceOp
from .build_kernels import build
from .all_gather import all_gather_2D
from .reduce_scatter import reduce_scatter_2D
import torch.distributed as dist
from mpi4py import MPI
import torch
from .process_groups import ProcessGroups

pg = None

def get_heir_pg():
    global pg 
    assert pg is not None, "did you call the patch_deepspeed function?"
    return pg

class dummy_request:
    def __init__(self):
        pass 

    def wait(self):
        pass


def is_global_pg(group):
    return (group is None) or (group == dist.group.WORLD) or (dist.get_world_size(group) == dist.get_world_size())

def patched_all_gather(self, output_tensor, input_tensor, group=None, async_op=False):
    if is_global_pg(group):
        all_gather_2D(output_tensor,
            input_tensor,
            group = get_heir_pg(),
            async_op = False,
            use_rd = True,
            use_yacl = True)
        if async_op:
            return dummy_request()
    else:
        return self.old_all_gather_into_tensor(output_tensor, input_tensor, group, async_op)

def patched_reduce_scatter(self, 
                           output_tensor, 
                           input_tensor, 
                           op=dist.ReduceOp.SUM, 
                           group=None, 
                           async_op=False):
    if is_global_pg(group) and (op == ReduceOp.SUM):
        
        reduce_scatter_2D(output_tensor,
                            input_tensor,
                            group = get_heir_pg(),
                            async_op = False,
                            use_rh = True,
                            use_yacl = True)
        if async_op:
            return dummy_request()
    else:
        return self.old_reduce_scatter_tensor(output_tensor, input_tensor, op, group, async_op)
    
def patched_all_reduce(self, tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if is_global_pg(group) and op == ReduceOp.SUM:
        output = torch.empty_like(tensor)
        torch.cuda.current_stream().synchronize()
        MPI.COMM_WORLD.Allreduce(tensor, output)
        tensor.copy_(output)
        if async_op:
            return dummy_request()
    else:
        return self.old_all_reduce(tensor, op, group, async_op)

def patched_barrier(self, group=torch.distributed.GroupMember.WORLD, async_op=False, device_ids=None):
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    if async_op:
        return dummy_request()
    
def patched_broadcast(self, tensor, src, group=None, async_op=False):
    if is_global_pg(group):
        MPI.COMM_WORLD.Bcast(tensor, root=src)
        if async_op:
            return dummy_request()
    else:
        return self.old_broadcast(tensor, src, group, async_op)


def patch_deepspeed(intra_node_pg_size):
    assert dist.is_initialized()
    assert dist.get_world_size() % intra_node_pg_size == 0
    if dist.get_rank() == 0:
        build()
        MPI.COMM_WORLD.Barrier()
    else:
        MPI.COMM_WORLD.Barrier()
        build()
    global pg 
    pg = ProcessGroups(intra_node_pg_size, 
                       dist.get_world_size() // intra_node_pg_size)

    #all-gather
    TorchBackend.old_all_gather_into_tensor = TorchBackend.all_gather_into_tensor
    TorchBackend.all_gather_into_tensor = patched_all_gather

    #reduce-scatter
    TorchBackend.old_reduce_scatter_tensor = TorchBackend.reduce_scatter_tensor
    TorchBackend.reduce_scatter_tensor = patched_reduce_scatter

    TorchBackend.old_all_reduce = TorchBackend.all_reduce 
    TorchBackend.all_reduce = patched_all_reduce

    TorchBackend.old_broadcast = TorchBackend.broadcast 
    TorchBackend.broadcast = patched_broadcast




