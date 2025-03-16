#include "reduce_scatter.h"
#include "utils.h"
#include <torch/extension.h>
#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <ATen/hip/HIPContext.h>

// Performs a recursive-halving reduce-scatter on GPU tensors.
//  - output: HIP device pointer where the reduced block (of block_size elements) will be stored.
//  - input: HIP device pointer to a contiguous buffer of (P * block_size) elements.
//  - total_elems: total number of elements (must equal P * block_size).
//  - comm: MPI communicator (default MPI_COMM_WORLD).
// The reduction operation is elementwise addition.
void recursiveHalvingReduceScatterGPU(float* output, 
    const float* input, 
    int total_elems,
    float* buf, // same as size of input 
    float* recv_buf, // same as size of input 
    MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;
    auto hip_stream = at::hip::getCurrentHIPStream();

    // copy the input into buf
    HIP_CHECK(hipMemcpyAsync(buf, 
        input, 
        total_elems * sizeof(float), 
        hipMemcpyDeviceToDevice,
        hip_stream));

    
    int max_count = (size / 2) * block_size;
    
    // Number of rounds = log2(size); assumes size is a power of 2.
    int rounds = static_cast<int>(std::log2(size));
    int current_blocks = size;
    // curr_ptr points to the beginning of the "active" region within buf.
    float* curr_ptr = buf;

    hipEvent_t stream_sync_event;
    HIP_CHECK(hipEventCreateWithFlags(&stream_sync_event, hipEventDisableTiming));

    for (int r = 0; r < rounds; ++r) {
        // In round r, the group size is:
        int group_size = size / (1 << r);
        // The current buffer holds 'current_blocks' contiguous blocks.
        int half = current_blocks / 2;
        // Number of elements to send/receive in this round.
        int count = half * block_size;

        if ((rank % group_size) < (group_size / 2)) {
            // Lower half: keep the lower half and send the upper half.
            int partner = (rank + half) % size;
            float* send_ptr = curr_ptr + half * block_size;
            // Record an event on the hip stream.
            HIP_CHECK(hipEventRecord(stream_sync_event, hip_stream));
            // Wait for the copy to complete.
            HIP_CHECK(hipEventSynchronize(stream_sync_event));

            MPI_Sendrecv(send_ptr, count, MPI_FLOAT, partner, r,
                         recv_buf, count, MPI_FLOAT, partner, r,
                         comm, MPI_STATUS_IGNORE);
            // Reduce the received data into the lower half.
            vectorAdd(curr_ptr, recv_buf, count, hip_stream);
            current_blocks = half;
        } else {
            // Upper half: keep the upper half and send the lower half.
            int partner = (rank - half + size) % size;
            float* send_ptr = curr_ptr;  // lower half to be sent.
            // Record an event on the hip stream.
            HIP_CHECK(hipEventRecord(stream_sync_event, hip_stream));
            // Wait for the copy to complete.
            HIP_CHECK(hipEventSynchronize(stream_sync_event));
            MPI_Sendrecv(send_ptr, count, MPI_FLOAT, partner, r,
                         recv_buf, count, MPI_FLOAT, partner, r,
                         comm, MPI_STATUS_IGNORE);
            float* kept_ptr = curr_ptr + half * block_size;
            vectorAdd(kept_ptr, recv_buf, count, hip_stream);
            curr_ptr = kept_ptr;
            current_blocks = half;
        }
    }

    // After log2(size) rounds, curr_ptr points to a single block (of block_size elements).
    HIP_CHECK(hipMemcpyAsync(output, 
        curr_ptr, 
        block_size * sizeof(float), 
        hipMemcpyDeviceToDevice, 
        hip_stream));

    HIP_CHECK(hipEventDestroy(stream_sync_event));
}

void ringReduceScatterGPU(float* output, 
    const float* input, 
    int total_elems, 
    float* d_buf, // same as size of input
    float* d_send, // same as size of output
    float* d_tmp, // same as size of output
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto hip_stream = at::hip::getCurrentHIPStream();
    hipEvent_t stream_sync_event;
    
    HIP_CHECK(hipMemcpyAsync(d_buf, 
        input, 
        total_elems * sizeof(float), 
        hipMemcpyDeviceToDevice,
        hip_stream));

    HIP_CHECK(hipEventCreateWithFlags(&stream_sync_event, hipEventDisableTiming));

    for (int step = 0; step < size - 1; step++) {
        // Compute block indices.
        int send_idx = (rank - step - 1 + size) % size;
        int recv_idx = (rank - step - 2 + size) % size;
        int dest     = (rank + 1) % size;
        int source   = (rank - 1 + size) % size;

        // Record an event on the hip stream.
        HIP_CHECK(hipEventRecord(stream_sync_event, hip_stream));
        // Wait for the copy to complete.
        HIP_CHECK(hipEventSynchronize(stream_sync_event));

        // Send the block to the right neighbor and receive from the left neighbor.
        MPI_Sendrecv(d_buf + send_idx * block_size, block_size, MPI_FLOAT, dest, 0,
                    d_tmp,  block_size, MPI_FLOAT, source, 0,
                    comm, MPI_STATUS_IGNORE);

        // Reduce the received block into the block at recv_idx.
        vectorAdd(d_buf + recv_idx * block_size, d_tmp, block_size, hip_stream);
    }
     
    HIP_CHECK(hipMemcpyAsync(output, 
        d_buf + rank * block_size, 
        block_size * sizeof(float), 
        hipMemcpyDeviceToDevice,
        hip_stream));

    HIP_CHECK(hipEventDestroy(stream_sync_event));
}

