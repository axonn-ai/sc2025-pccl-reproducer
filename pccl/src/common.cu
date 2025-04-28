#include "common.h"


// Kernel for vector addition.
__global__ void vectorAddKernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

// Function to launch the kernel.
void vectorAdd(float* a, const float* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
   vectorAddKernel<<<blocks, threads, 0, stream>>>(a, b, n);
   CUDA_CHECK(cudaGetLastError());
}

