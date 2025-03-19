#ifndef UTILS_H
#define UTILS_H

#include <hip/hip_runtime.h>
#include <iostream>

// Utility macro for HIP error checking.
#define HIP_CHECK(call) do { \
  hipError_t err = call; \
  if(err != hipSuccess) { \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                << " code=" << err << " \"" << hipGetErrorString(err) << "\"\n"; \
      exit(1); \
  } } while(0)

// HIP kernel for vector addition.
__global__ void vectorAddKernel(float* a, const float* b, int n);

// Function to launch HIP kernel.
void vectorAdd(float* a, const float* b, int n, hipStream_t stream);

#endif // UTILS_H
