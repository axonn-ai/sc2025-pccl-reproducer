import torch
import os 
import torch.distributed as dist
from torch.profiler import record_function

def init():
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    dist.init_process_group(rank=rank, 
                            world_size=world_size,
                            backend="nccl", 
                            init_method="env://")
    torch.cuda.set_device(rank %  torch.cuda.device_count())
    

def time_something(fn, *args, warmup_iters=5, timed_iters=20, prof=None, **kwargs):
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

def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True
