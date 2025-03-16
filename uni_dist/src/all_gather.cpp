#include "all_gather.h"
#include "utils.h"
#include <torch/extension.h>
#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <ATen/hip/HIPContext.h>

// Performs a recursive doubling all-gather on GPU tensors.
//  - output: HIP device pointer where the final gathered tensor will be stored.
//  - input: HIP device pointer to the local block of size block_size.
//  - total_elems: total number of elements in output (P * block_size).
//  - comm: MPI communicator (default MPI_COMM_WORLD).
void recursiveDoublingAllGatherGPU(void* output, 
                                  const void* input, 
                                  int total_elems, 
                                  void* recv_buf,  // Same as output size
                                  MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;
    auto hip_stream = at::hip::getCurrentHIPStream();

    // Copy local input into its designated block in the output buffer.
    HIP_CHECK(hipMemcpyAsync(static_cast<char*>(output) + rank * block_size, 
                             input, 
                             block_size, 
                             hipMemcpyDeviceToDevice, 
                             hip_stream));

    hipEvent_t stream_sync_event;
    HIP_CHECK(hipEventCreateWithFlags(&stream_sync_event, hipEventDisableTiming));

    int seg_size = 1;  // Start with local block.
    while (seg_size < size) {
        int partner = rank ^ seg_size;
        int group_start = (rank / (2 * seg_size)) * (2 * seg_size);
        int send_offset, recv_offset;
        
        if (rank < partner) {
            send_offset = group_start * block_size;
            recv_offset = (group_start + seg_size) * block_size;
        } else {
            send_offset = (group_start + seg_size) * block_size;
            recv_offset = group_start * block_size;
        }

        int count = seg_size * block_size;
        
        // Record an event on the hip stream.
        HIP_CHECK(hipEventRecord(stream_sync_event, hip_stream));
        // Wait for the copy to complete.
        HIP_CHECK(hipEventSynchronize(stream_sync_event));
        
        MPI_Sendrecv(static_cast<char*>(output) + send_offset, count, MPI_BYTE, partner, 0,
                     static_cast<char*>(output) + recv_offset, count, MPI_BYTE, partner, 0,
                     comm, MPI_STATUS_IGNORE);
        
        // HIP_CHECK(hipMemcpyAsync(static_cast<char*>(output) + recv_offset, 
        //                          recv_buf, 
        //                          count, 
        //                          hipMemcpyDeviceToDevice, 
        //                          hip_stream));
        
        seg_size *= 2;
    }
}
