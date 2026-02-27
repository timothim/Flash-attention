/**
 * naive_attention.cu — Baseline attention that materialises the full N×N
 * score matrix.  This is intentionally simple and serves as a correctness
 * reference and performance baseline.
 *
 * Computes:   S = Q × K^T × scale          (B, H, N, N)
 *             P = softmax(S, dim=-1)        (B, H, N, N)  — with optional causal mask
 *             O = P × V                     (B, H, N, D)
 *
 * Memory:  O(B H N²) — quadratic in seq_len.
 */

#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <cfloat>

// ════════════════════════════════════════════════════════════════════════
// Kernel 1:  S = Q × K^T × scale  (with optional causal mask)
// ════════════════════════════════════════════════════════════════════════
//
// Grid  : (cdiv(N, TILE), cdiv(N, TILE), B * H)
// Block : (TILE, TILE)  — each thread computes one element of S.

static constexpr int NAIVE_TILE = 16;

__global__ void naive_qk_kernel(
    const half* __restrict__ Q,   // [B*H, N, D]
    const half* __restrict__ K,   // [B*H, N, D]
    float*      __restrict__ S,   // [B*H, N, N]
    int N, int D, float scale, bool causal)
{
    const int bh = blockIdx.z;
    const int row = blockIdx.x * NAIVE_TILE + threadIdx.x;
    const int col = blockIdx.y * NAIVE_TILE + threadIdx.y;
    if (row >= N || col >= N) return;

    // Causal mask: positions where col > row are masked to -inf.
    if (causal && col > row) {
        S[bh * N * N + row * N + col] = -1e20f;
        return;
    }

    const half* q_row = Q + bh * N * D + row * D;
    const half* k_row = K + bh * N * D + col * D;

    float acc = 0.0f;
    for (int d = 0; d < D; ++d)
        acc += __half2float(q_row[d]) * __half2float(k_row[d]);

    S[bh * N * N + row * N + col] = acc * scale;
}

// ════════════════════════════════════════════════════════════════════════
// Kernel 2:  Row-wise softmax  P = softmax(S, dim=-1)
// ════════════════════════════════════════════════════════════════════════
//
// Grid  : (N, B * H)
// Block : (256)
// Each block handles one row of N elements (loop if N > 256).

__global__ void naive_softmax_kernel(
    const float* __restrict__ S,   // [B*H, N, N]
    float*       __restrict__ P,   // [B*H, N, N]
    int N)
{
    const int bh  = blockIdx.y;
    const int row = blockIdx.x;
    if (row >= N) return;

    const float* s_row = S + bh * N * N + row * N;
    float*       p_row = P + bh * N * N + row * N;

    // --- pass 1: find row max ---
    extern __shared__ float sdata[];
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        local_max = fmaxf(local_max, s_row[j]);

    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // --- pass 2: compute exp and sum ---
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = __expf(s_row[j] - row_max);
        p_row[j] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sdata[0];

    // --- pass 3: normalise ---
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        p_row[j] *= inv_sum;
}

// ════════════════════════════════════════════════════════════════════════
// Kernel 3:  O = P × V
// ════════════════════════════════════════════════════════════════════════
//
// Grid  : (cdiv(N, TILE), cdiv(D, TILE), B * H)
// Block : (TILE, TILE)

__global__ void naive_pv_kernel(
    const float* __restrict__ P,   // [B*H, N, N]
    const half*  __restrict__ V,   // [B*H, N, D]
    half*        __restrict__ O,   // [B*H, N, D]
    int N, int D)
{
    const int bh  = blockIdx.z;
    const int row = blockIdx.x * NAIVE_TILE + threadIdx.x;
    const int col = blockIdx.y * NAIVE_TILE + threadIdx.y;
    if (row >= N || col >= D) return;

    const float* p_row = P + bh * N * N + row * N;
    const half*  v_col = V + bh * N * D;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j)
        acc += p_row[j] * __half2float(v_col[j * D + col]);

    O[bh * N * D + row * D + col] = __float2half(acc);
}

// ════════════════════════════════════════════════════════════════════════
// Host launcher
// ════════════════════════════════════════════════════════════════════════

void naive_attention_fwd(
    const half* Q,    // [B*H, N, D]  device ptr
    const half* K,
    const half* V,
    half*       O,    // [B*H, N, D]
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream)
{
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    // Allocate temporaries S, P on device (O(B H N²) — intentionally wasteful).
    float *d_S = nullptr, *d_P = nullptr;
    size_t sn_bytes = (size_t)BH * N * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_S, sn_bytes));
    CUDA_CHECK(cudaMalloc(&d_P, sn_bytes));

    // --- S = Q × K^T × scale ---
    {
        dim3 block(NAIVE_TILE, NAIVE_TILE);
        dim3 grid(cdiv(N, NAIVE_TILE), cdiv(N, NAIVE_TILE), BH);
        naive_qk_kernel<<<grid, block, 0, stream>>>(Q, K, d_S, N, D, scale, causal);
    }

    // --- P = softmax(S) ---
    {
        int threads = 256;
        dim3 grid(N, BH);
        int smem = threads * sizeof(float);
        naive_softmax_kernel<<<grid, threads, smem, stream>>>(d_S, d_P, N);
    }

    // --- O = P × V ---
    {
        dim3 block(NAIVE_TILE, NAIVE_TILE);
        dim3 grid(cdiv(N, NAIVE_TILE), cdiv(D, NAIVE_TILE), BH);
        naive_pv_kernel<<<grid, block, 0, stream>>>(d_P, V, O, N, D);
    }

    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
}
