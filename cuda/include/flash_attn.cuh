#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ════════════════════════════════════════════════════════════════════════
// Flash Attention 2 — Public kernel launch API
//
// All tensors are expected in [B*H, N, D] row-major layout (batch and
// head dimensions fused).  The caller is responsible for reshaping from
// [B, H, N, D] if needed — this is trivial since the memory is
// contiguous.
//
// Half precision (FP16) is used for Q, K, V, O, dQ, dK, dV.
// Float32 is used for the logsumexp vector L.
// ════════════════════════════════════════════════════════════════════════

/// Naive (materialising) attention — baseline for correctness / benchmarks.
void naive_attention_fwd(
    const half* Q,    // [B*H, N, D]
    const half* K,
    const half* V,
    half*       O,
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream = 0);

/// Flash Attention 2 — forward pass.
///   O = softmax(Q K^T / sqrt(d)) V
///   L = logsumexp of each row (saved for backward)
void flash_attn_fwd(
    const half* Q,
    const half* K,
    const half* V,
    half*       O,
    float*      L,    // [B*H, N]
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream = 0);

/// Flash Attention 2 — backward pass.
///   Computes dQ, dK, dV given the saved tensors from forward and dO.
void flash_attn_bwd(
    const half*  Q,
    const half*  K,
    const half*  V,
    const half*  O,
    const half*  dO,
    const float* L,
    half*        dQ,
    half*        dK,
    half*        dV,
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream = 0);
