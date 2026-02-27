#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"

// ════════════════════════════════════════════════════════════════════════
// Shared memory helpers
//
// Shared memory on NVIDIA GPUs is divided into 32 banks, each 4 bytes
// wide.  When threads in a warp access the same bank (but different
// addresses) the accesses serialise — a "bank conflict".
//
// For a row-major half-precision matrix K_smem[Bc][D]:
//   stride = D halfs = D * 2 bytes.
//   If D = 64  → stride = 128 bytes = 32 banks exactly → column
//   accesses cause 32-way conflicts.
//
// Fix: pad each row by PAD_HALFS so the stride is no longer a multiple
// of 32 banks.  PAD = 8 halfs (16 bytes) is a standard choice.
// ════════════════════════════════════════════════════════════════════════

constexpr int SMEM_PAD_HALFS = 8;  // 8 × sizeof(half) = 16 bytes

/// Padded stride (in half elements) for a row of width `cols`.
__host__ __device__ __forceinline__
constexpr int padded_stride(int cols) { return cols + SMEM_PAD_HALFS; }

/// Total shared-memory bytes for a padded [rows × cols] half matrix.
__host__ __device__ __forceinline__
constexpr int smem_padded_bytes(int rows, int cols) {
    return rows * padded_stride(cols) * sizeof(half);
}

// ════════════════════════════════════════════════════════════════════════
// Cooperative tile loads  (HBM → shared memory)
//
// Each thread in the block loads a few elements.  We assume row-major
// layout both in global and shared memory (with padding in smem).
// ════════════════════════════════════════════════════════════════════════

/// Load a [rows × cols] tile of half values from global memory into
/// padded shared memory.  Works for any block size.
///
///   dst       — pointer into __shared__ half array (padded layout)
///   src       — pointer into global half array (row-major, stride = cols)
///   rows,cols — logical tile shape
///   tid       — threadIdx.x
///   n_threads — blockDim.x
///   max_row   — clamp: only load rows < max_row (for boundary tiles)
///
/// Elements beyond max_row are zero-filled.
__device__ __forceinline__
void smem_load_tile(half* __restrict__ dst,
                    const half* __restrict__ src,
                    int rows, int cols,
                    int tid, int n_threads,
                    int max_row)
{
    const int stride_dst = padded_stride(cols);
    const int total = rows * cols;
    for (int idx = tid; idx < total; idx += n_threads) {
        int r = idx / cols;
        int c = idx % cols;
        half val = __float2half(0.0f);
        if (r < max_row) {
            val = src[r * cols + c];
        }
        dst[r * stride_dst + c] = val;
    }
}

/// Same as above but source pointer has a custom stride (for non-
/// contiguous head_dim layouts).
__device__ __forceinline__
void smem_load_tile_strided(half* __restrict__ dst,
                            const half* __restrict__ src,
                            int rows, int cols,
                            int src_row_stride,
                            int tid, int n_threads,
                            int max_row)
{
    const int stride_dst = padded_stride(cols);
    const int total = rows * cols;
    for (int idx = tid; idx < total; idx += n_threads) {
        int r = idx / cols;
        int c = idx % cols;
        half val = __float2half(0.0f);
        if (r < max_row) {
            val = src[r * src_row_stride + c];
        }
        dst[r * stride_dst + c] = val;
    }
}

/// Store a [rows × cols] tile from padded shared memory back to global
/// memory (row-major, stride = cols).
__device__ __forceinline__
void smem_store_tile(half* __restrict__ dst,
                     const half* __restrict__ src,
                     int rows, int cols,
                     int tid, int n_threads,
                     int max_row)
{
    const int stride_src = padded_stride(cols);
    const int total = rows * cols;
    for (int idx = tid; idx < total; idx += n_threads) {
        int r = idx / cols;
        int c = idx % cols;
        if (r < max_row) {
            dst[r * cols + c] = src[r * stride_src + c];
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Float <-> half conversion helpers in registers
// ════════════════════════════════════════════════════════════════════════

__device__ __forceinline__
float h2f(half v) { return __half2float(v); }

__device__ __forceinline__
half f2h(float v) { return __float2half(v); }
