import torch
import torch.distributed as dist
from mpi4py import MPI
import os
from linear import time_something, init
from argparse import ArgumentParser


def send_recv(buffer, send_rank, recv_rank):
    if dist.get_rank() == send_rank:
        req = MPI.COMM_WORLD.Isend(buffer, recv_rank, 0)
        req.Wait()
        req = MPI.COMM_WORLD.Irecv(buffer, recv_rank, 1)
        req.Wait()
    elif dist.get_rank() == recv_rank:
        req = MPI.COMM_WORLD.Irecv(buffer, send_rank, 0)
        req.Wait()
        req = MPI.COMM_WORLD.Isend(buffer, send_rank, 1)
        req.Wait()


def send_recv_multi_nic(buffer, intra_node_group, inter_node_group):
    intermediate_buffer = torch.empty(
        (buffer.size(0) // intra_node_group.Get_size(),),
        device=buffer.device,
        dtype=buffer.dtype,
    )
    if inter_node_group.Get_rank() == 0:  # sender processes
        ## Send
        req = intra_node_group.Iscatterv(
            buffer, intermediate_buffer, root=inter_node_group.Get_size() - 1
        )
        req.Wait()
        req = inter_node_group.Isend(intermediate_buffer, dest=1, tag=0)
        req.Wait()

        ## Recv
        req = inter_node_group.Irecv(intermediate_buffer, source=1, tag=1)
        req.Wait()
        req = intra_node_group.Igatherv(intermediate_buffer, buffer, root=0)
        req.Wait()
    else:
        ## Recv
        req = inter_node_group.Irecv(intermediate_buffer, source=0, tag=0)
        req.Wait()
        req = intra_node_group.Igatherv(intermediate_buffer, buffer, root=0)
        req.Wait()

        ## Send
        req = intra_node_group.Iscatterv(buffer, intermediate_buffer, root=0)
        req.Wait()
        req = inter_node_group.Isend(intermediate_buffer, dest=0, tag=1)
        req.Wait()


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

    intra_node_group = MPI.COMM_WORLD.Split(dist.get_rank() // gpus_per_node)
    inter_node_group = MPI.COMM_WORLD.Split(dist.get_rank() % gpus_per_node)

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
