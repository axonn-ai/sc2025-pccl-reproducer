import torch 
import torch.nn as nn
import torch.distributed as dist
from all_gather import all_gather 
from reduce_scatter import reduce_scatter
from uni_dist.process_groups import create_2D_grid
from torch.autograd import Function
import torch.nn.functional as F
import os
import math
import random 
import numpy as np
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from torch.profiler import _KinetoProfile
_KinetoProfile._get_distributed_info = lambda self: None

def div(a, b):
    assert a%b == 0
    return a//b

class AG_RS(Function):
    @staticmethod
    def forward(ctx, msg, process_groups, comm_lib):
        output = all_gather(msg, process_groups, comm_lib)
        ctx.comm_lib = comm_lib
        ctx.process_groups = process_groups 
        return output

    @staticmethod 
    def backward(ctx, grad_output):
        grad_input = reduce_scatter(grad_output, ctx.process_groups, ctx.comm_lib)
        return grad_input, None, None

class RS_AG(Function):
    @staticmethod
    def forward(ctx, msg, process_groups, comm_lib):
        output = reduce_scatter(msg, process_groups, comm_lib)
        ctx.comm_lib = comm_lib
        ctx.process_groups = process_groups 
        return output

    @staticmethod 
    def backward(ctx, grad_output):
        grad_input = all_gather(grad_output, ctx.process_groups, ctx.comm_lib)
        return grad_input, None, None

class TPLinear(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 mode,
                 comm_lib,
                 process_groups):
        assert mode in ["row", "col"]
        assert comm_lib in ["hybrid", "nccl"]
        super(TPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode 
        if not isinstance(process_groups, tuple):
            process_groups = (process_groups,)
        self.process_groups = process_groups
        self.comm_lib = comm_lib
        if comm_lib == "nccl":
            assert len(process_groups) == 1 
            self.world_size = dist.get_world_size(process_groups[0])
        elif comm_lib == "hybrid":
            assert len(process_groups) == 2
            self.world_size = dist.get_world_size(process_groups[0]) * process_groups[1].Get_size()
        

        self.local_out_features = div(out_features, self.world_size) if mode == "col" else out_features
        self.local_in_features = div(in_features, self.world_size) if mode == "row" else in_features 

        weight = nn.Parameter(torch.Tensor(self.local_out_features, self.local_in_features))
        self.weight = nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


    def forward(self, x):
        if self.mode == "col":
            # all-gather input 
            x = AG_RS.apply(x.view(-1), self.process_groups, self.comm_lib).view(-1, x.size(1), x.size(2))
            return F.linear(x, self.weight)
        elif self.mode == "row":
            x = F.Linear(x, self.weight)
            return RS_AG.apply(x.view(-1), self.process_groups, self.comm_lib).view(-1, x.size(1), x.size(2))
        

def init():
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    dist.init_process_group(rank=rank, 
                            world_size=world_size,
                            backend="nccl", 
                            init_method="env://")
    torch.cuda.set_device(rank %  torch.cuda.device_count())
    

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def allclose(x, y, pct=2.0):
    # taken from https://github.com/tgale96/grouped_gemm/blob/main/grouped_gemm/ops_test.py
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True

def time_something(fn, *args, warmup_iters=5, timed_iters=10, prof=None, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True) 
    end_event = torch.cuda.Event(enable_timing=True) 
    for i in range(warmup_iters):
        with record_function(f"warmup_{i}"):
            fn(*args, **kwargs)
        #if prof is not None:
        #prof.step()
    torch.cuda.synchronize()
    start_event.record()
    for i in range(timed_iters):
        with record_function(f"iteration_{i}"):
            fn(*args, **kwargs)
        #if prof is not None:
        #prof.step()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / timed_iters


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


if __name__ == "__main__":
    mbs = 1
    sq = 256
    hin = 8192 
    hout = 28672
    init()
    set_seed(1234+ dist.get_rank())
    hybrid_pgs = create_2D_grid(4, dist.get_world_size() // 4)

    layer_nccl = TPLinear(hin, hout, "col", "nccl", None).cuda().to(torch.bfloat16)
    layer_hybrid = TPLinear(hin, hout, "col", "hybrid", hybrid_pgs).cuda().to(torch.bfloat16)
    layer_hybrid.weight.data.copy_(layer_nccl.weight.data)

    x = torch.randn(mbs, sq // dist.get_world_size(), hin).cuda().to(torch.bfloat16)

    y_nccl = layer_nccl(x)
    y_hybrid = layer_hybrid(x)

    assert allclose(y_nccl, y_hybrid)
    print("correctness test passed in fw pass!")
    
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p1: 
            #schedule=schedule(wait=1, warmup=4, active=2)) as prof_:
    nccl_time = time_something(layer_nccl, x)
    #if dist.get_rank() == 0:
    #    p1.export_chrome_trace("nccl.json")
        
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p2: 
    #        #schedule=schedule(wait=1, warmup=4, active=2)) as prof:
    hybrid_time = time_something(layer_hybrid, x)

    #if dist.get_rank() == 0:
    #    p2.export_chrome_trace("hybrid.json")

    print_rank0("NCCL time: {:.4f} ms".format(nccl_time))
    print_rank0("Hybrid time: {:.4f} ms".format(hybrid_time))