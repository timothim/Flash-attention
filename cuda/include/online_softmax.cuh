#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

// ════════════════════════════════════════════════════════════════════════
// Warp-level reductions
//
// Each warp (32 threads) can communicate via __shfl_xor_sync without
// touching shared memory.  A butterfly pattern performs the reduction
// in log2(32) = 5 steps.
// ════════════════════════════════════════════════════════════════════════

__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ════════════════════════════════════════════════════════════════════════
// Cross-warp reductions via shared memory
//
// When a single logical row is distributed across more than one warp
// (e.g. Bc = 64 with 32 threads/warp → 2 warps per row), we first
// reduce within each warp, then use shared memory to combine.
// ════════════════════════════════════════════════════════════════════════

/// Reduce-max across `n_warps_per_row` warps.
/// `smem_reduce` must have room for at least n_warps_per_row floats.
__device__ __forceinline__
float block_reduce_max(float val, float* smem_reduce,
                       int lane, int warp_id, int n_warps_per_row)
{
    val = warp_reduce_max(val);
    if (lane == 0)
        smem_reduce[warp_id] = val;
    __syncthreads();
    if (warp_id == 0 && lane < n_warps_per_row)
        val = smem_reduce[lane];
    else if (warp_id == 0)
        val = -FLT_MAX;
    if (warp_id == 0)
        val = warp_reduce_max(val);
    // Broadcast the result back via smem.
    if (warp_id == 0 && lane == 0)
        smem_reduce[0] = val;
    __syncthreads();
    return smem_reduce[0];
}

/// Reduce-sum across `n_warps_per_row` warps.
__device__ __forceinline__
float block_reduce_sum(float val, float* smem_reduce,
                       int lane, int warp_id, int n_warps_per_row)
{
    val = warp_reduce_sum(val);
    if (lane == 0)
        smem_reduce[warp_id] = val;
    __syncthreads();
    if (warp_id == 0 && lane < n_warps_per_row)
        val = smem_reduce[lane];
    else if (warp_id == 0)
        val = 0.0f;
    if (warp_id == 0)
        val = warp_reduce_sum(val);
    if (warp_id == 0 && lane == 0)
        smem_reduce[0] = val;
    __syncthreads();
    return smem_reduce[0];
}

// ════════════════════════════════════════════════════════════════════════
// Online softmax state
//
// Maintains per-row running statistics (max, sum-of-exp) so that the
// final softmax can be computed incrementally as new KV blocks arrive,
// without ever materialising the full N×N score matrix.
//
// Reference: Milakov & Gimelshein, "Online normalizer calculation for
//            softmax", arXiv 1805.02867 (2018).
// ════════════════════════════════════════════════════════════════════════

struct OnlineSoftmaxState {
    float m;   // running max
    float l;   // running sum of exp(x - m)

    __device__ __forceinline__
    OnlineSoftmaxState() : m(-FLT_MAX), l(0.0f) {}

    /// Absorb a new block of scores.  Returns the rescaling factor alpha
    /// that must be applied to the *previous* accumulator O_acc.
    ///
    ///   new_max  — max over the current S_block row
    ///   new_sum  — sum of exp(S_block[row] - new_max) for the current block
    ///
    /// After calling update():
    ///   m = max(old_m, new_max)
    ///   l = exp(old_m - m) * old_l + exp(new_max - m) * new_sum
    ///   alpha = exp(old_m - m)   (rescale factor for O_acc)
    __device__ __forceinline__
    float update(float new_max, float new_sum) {
        float m_new   = fmaxf(m, new_max);
        float alpha   = __expf(m - m_new);
        float beta    = __expf(new_max - m_new);
        l = alpha * l + beta * new_sum;
        m = m_new;
        return alpha;
    }

    /// Compute the log-sum-exp (saved for backward pass).
    __device__ __forceinline__
    float logsumexp() const {
        return m + logf(l);
    }
};
