#ifndef COMMON_H
#define COMMON_H

#if defined(USE_CUDA)
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  // Type aliases.
  typedef cudaStream_t hipStream_t;
  typedef cudaEvent_t  hipEvent_t;
  
  // Function name aliases.
  #define hipMemcpyAsync        cudaMemcpyAsync
  #define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
  #define hipEventCreateWithFlags(cudaEvent, flags) cudaEventCreateWithFlags(cudaEvent, flags)
  #define hipEventRecord        cudaEventRecord
  #define hipEventSynchronize   cudaEventSynchronize
  #define hipGetErrorString     cudaGetErrorString
  #define hipGetLastError      cudaGetLastError
  #define hipEventDestroy       cudaEventDestroy
  #define hipEventDisableTiming cudaEventDisableTiming

  // Macro for error checking.
  #define HIP_CHECK(call) do { \
      cudaError_t err = call; \
      if(err != cudaSuccess) { \
          std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                    << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
          exit(1); \
      } \
  } while(0)
#elif defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <ATen/hip/HIPContext.h>

  // Macro for error checking.
  #define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if(err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << err << " \"" << hipGetErrorString(err) << "\"\n"; \
        exit(1); \
    } \

  } while(0)
#else
  #error "Either USE_CUDA or USE_ROCM must be defined"
#endif

#include <iostream>

// Kernel for vector addition.
__global__ void vectorAddKernel(float* a, const float* b, int n);

// Function to launch the kernel.
void vectorAdd(float* a, const float* b, int n, hipStream_t stream);

#endif // COMMON_H
