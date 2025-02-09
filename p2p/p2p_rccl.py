import torch
import torch.distributed as dist
from mpi4py import MPI
import os
from linear import time_something, init
from argparse import ArgumentParser
import numpy as np


def send_recv(buffer, send_rank, recv_rank):
    if dist.get_rank() == send_rank:
        req = dist.isend(buffer, recv_rank, tag=0)
        req.wait()
        req = dist.irecv(buffer, recv_rank, tag=1)
        req.wait()
    elif dist.get_rank() == recv_rank:
        req = dist.irecv(buffer, send_rank, tag=0)
        req.wait()
        req = dist.isend(buffer, send_rank, tag=1)
        req.wait()


def _scatter(send_buff, recv_buff, group, root, async_op=False):
    buffer_list = torch.split(send_buff, buffer.size(0) // dist.get_world_size(group))
    is_root = (dist.get_rank(group) == root)
    req = dist.scatter(recv_buff, 
                       scatter_list=list(buffer_list) if is_root else None, 
                       group_src=root, 
                       async_op=async_op, group=group)
    return req

def _gather(send_buff, recv_buff, group, root, async_op=False):
    buffer_list = torch.split(recv_buff, buffer.size(0) // dist.get_world_size(group))
    is_root = (dist.get_rank(group) == root)
    req = dist.gather(send_buff, 
                      gather_list=list(buffer_list) if is_root else None, 
                      group_dst=root, 
                      async_op=async_op, 
                      group=group)
    return req

def send_recv_multi_nic(buffer, intra_node_group, inter_node_group):
    intra_node_group_size = dist.get_world_size(intra_node_group)
    inter_node_group_size = dist.get_world_size(inter_node_group)
    intermediate_buffer = torch.empty(
        (buffer.size(0) // intra_node_group_size,),
        device=buffer.device,
        dtype=buffer.dtype,
    )
    if dist.get_rank(inter_node_group) == 0:  # sender processes
        ## Send
        req = _scatter(
            buffer, intermediate_buffer, root=intra_node_group_size-1, group=intra_node_group, async_op=True
        )
        req.wait()
        req = dist.isend(intermediate_buffer, group_dst=1, tag=0, group=inter_node_group)
        req.wait()

        ## Recv
        req = dist.irecv(intermediate_buffer, group_src=1, tag=1, group=inter_node_group)
        req.wait() 
        req = _gather(
            intermediate_buffer, buffer, root=0, group=intra_node_group, async_op=True
        )

    else:
        ## Recv
        req = dist.irecv(intermediate_buffer, group_src=0, tag=0, group=inter_node_group)
        req.wait()
        req = _gather(
            intermediate_buffer, buffer, root=0, group=intra_node_group, async_op=True
        )
        
        ## Send
        req = _scatter(
            buffer, intermediate_buffer, root=0, group=intra_node_group, async_op=True
        )
        req.wait()
        req = dist.isend(intermediate_buffer, group_dst=0, tag=1, group=inter_node_group)
        req.wait()

        


if __name__ == "__main__":
    init()
    parser = ArgumentParser()
    # Add these arguments to the parser - --optimized
    parser.add_argument("--optimized", action="store_true")
    args = parser.parse_args()

    gpus_per_node = 4
    # we are simulating the inter-node send/recv in pipelining
    # last gpu of node 0 <-> first gpu of node 1
    send_rank = gpus_per_node - 1
    recv_rank = gpus_per_node
    assert dist.get_world_size() > recv_rank

    # Create intra and inter node groups for nccl
    ranks = np.arange(dist.get_world_size()).reshape(2, gpus_per_node)
    for i in range(2):
        this_ranks = ranks[i].tolist()
        this_group = dist.new_group(ranks=this_ranks, backend="nccl")
        if dist.get_rank() in this_ranks:
            intra_node_group = this_group

    for i in range(gpus_per_node):
        this_ranks = ranks[:, i].tolist()
        this_group = dist.new_group(ranks=this_ranks, backend="nccl")
        if dist.get_rank() in this_ranks:
            inter_node_group = this_group

    if not args.optimized:
        if dist.get_rank() in [send_rank, recv_rank]:
            for data_size_mb in [16, 32, 64, 128, 256, 512, 1024]:
                buffer = torch.empty(
                    (data_size_mb * 2**20 // 2,), device="cuda", dtype=torch.bfloat16
                )
                time = time_something(send_recv, buffer, send_rank, recv_rank)
                bandwidth_gbps = 2 * data_size_mb / time
                if dist.get_rank() == recv_rank:
                    print(
                        f"size = {data_size_mb} MB, bandwidth = {bandwidth_gbps:.2f} GBPS"
                    )

    else:
        for data_size_mb in [16, 32, 64, 128, 256, 512, 1024]:
            buffer = torch.empty(
                (data_size_mb * 2**20 // 2,), device="cuda", dtype=torch.bfloat16
            )
            time = time_something(
                send_recv_multi_nic,
                buffer,
                intra_node_group,
                inter_node_group,
                timed_iters=10,
            )
            bandwidth_gbps = 2 *data_size_mb / time
            if dist.get_rank() == 0:
                print(
                    f"rank = {dist.get_rank()}, size = {data_size_mb} MB, bandwidth = {bandwidth_gbps:.2f} GBPS"
                )
