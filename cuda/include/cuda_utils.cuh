#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

// ════════════════════════════════════════════════════════════════════════
// Error checking
// ════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error at %s:%d — %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ════════════════════════════════════════════════════════════════════════
// Timing helpers (GPU events)
// ════════════════════════════════════════════════════════════════════════

struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    /// Returns elapsed time in milliseconds.
    float toc(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ════════════════════════════════════════════════════════════════════════
// Numeric helpers
// ════════════════════════════════════════════════════════════════════════

/// Ceiling division (positive integers).
__host__ __device__ __forceinline__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

/// Infinity constant for FP32 masking.
constexpr float NEG_INF = -1e20f;

// ════════════════════════════════════════════════════════════════════════
// Launch-config helpers
// ════════════════════════════════════════════════════════════════════════

/// Query and print the max shared-memory per block for the current device.
inline void print_smem_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_per_block, smem_per_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block,
               cudaDevAttrMaxSharedMemoryPerBlock, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm,
               cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    printf("Device %d — smem/block: %d KB, smem/SM: %d KB\n",
           device, smem_per_block / 1024, smem_per_sm / 1024);
}

/// Set the maximum dynamic shared memory for a kernel.
template <typename Kernel>
inline void set_smem_limit(Kernel kernel, int smem_bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
}
