/**
 * flash_attn_bwd.cu — Flash Attention 2 backward pass.
 *
 * The backward recomputes S and P from Q, K, V and the saved logsumexp L
 * instead of storing the full N×N matrices (O(N) memory vs O(N²)).
 *
 * Loop structure (reversed vs forward):
 *   Outer loop: over KV blocks   → accumulate dK, dV in registers
 *   Inner loop: over Q blocks    → load Q, dO, O, L from HBM / smem
 *
 * Gradients:
 *   dV += P^T @ dO
 *   dP  = dO @ V^T
 *   dS  = P * (dP - D_i)       where D_i = rowsum(dO * O)
 *   dQ += dS @ K / sqrt(d)     (accumulated via atomicAdd)
 *   dK += dS^T @ Q / sqrt(d)
 */

#include "cuda_utils.cuh"
#include "smem_utils.cuh"
#include "online_softmax.cuh"

static constexpr int BWD_BR = 64;   // Q block size
static constexpr int BWD_BC = 64;   // KV block size
static constexpr int BWD_NUM_THREADS = 128;

// ════════════════════════════════════════════════════════════════════════
// Pre-compute D_i = rowsum(dO * O) for every row.
// Grid: (cdiv(N, 256), B*H)      Block: 256
// ════════════════════════════════════════════════════════════════════════

__global__ void compute_D_kernel(
    const half*  __restrict__ dO,   // [B*H, N, D]
    const half*  __restrict__ O,    // [B*H, N, D]
    float*       __restrict__ D_vec,// [B*H, N]
    int N, int D_dim)
{
    const int bh  = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const half* do_row = dO + (size_t)bh * N * D_dim + row * D_dim;
    const half* o_row  = O  + (size_t)bh * N * D_dim + row * D_dim;

    float acc = 0.0f;
    for (int d = 0; d < D_dim; ++d)
        acc += __half2float(do_row[d]) * __half2float(o_row[d]);

    D_vec[bh * N + row] = acc;
}

// ════════════════════════════════════════════════════════════════════════
// Main backward kernel
//
// Grid  : (cdiv(N, BWD_BC), B*H)   — one block per KV chunk
// Block : (BWD_NUM_THREADS)
// ════════════════════════════════════════════════════════════════════════

__global__ void flash_attn_bwd_kernel(
    const half*  __restrict__ Q,      // [B*H, N, D]
    const half*  __restrict__ K,      // [B*H, N, D]
    const half*  __restrict__ V,      // [B*H, N, D]
    const half*  __restrict__ O,      // [B*H, N, D]
    const half*  __restrict__ dO,     // [B*H, N, D]
    const float* __restrict__ L,      // [B*H, N]
    const float* __restrict__ D_vec,  // [B*H, N]
    float*       __restrict__ dQ_f,   // [B*H, N, D]  float accum (atomics)
    half*        __restrict__ dK,     // [B*H, N, D]
    half*        __restrict__ dV,     // [B*H, N, D]
    int N, int D_dim, float scale, bool causal)
{
    const int bh        = blockIdx.y;
    const int kv_block  = blockIdx.x;
    const int kv_start  = kv_block * BWD_BC;
    const int kv_valid  = min(BWD_BC, N - kv_start);
    const int tid       = threadIdx.x;

    const half* Q_bh  = Q    + (size_t)bh * N * D_dim;
    const half* K_bh  = K    + (size_t)bh * N * D_dim;
    const half* V_bh  = V    + (size_t)bh * N * D_dim;
    const half* O_bh  = O    + (size_t)bh * N * D_dim;
    const half* dO_bh = dO   + (size_t)bh * N * D_dim;
    const float* L_bh = L    + (size_t)bh * N;
    const float* D_bh = D_vec + (size_t)bh * N;
    float* dQ_bh      = dQ_f + (size_t)bh * N * D_dim;
    half*  dK_bh      = dK   + (size_t)bh * N * D_dim;
    half*  dV_bh      = dV   + (size_t)bh * N * D_dim;

    const int d_padded = padded_stride(D_dim);

    // Shared memory: Q_smem, dO_smem for the inner Q-block loop.
    extern __shared__ char smem_raw[];
    half* Q_smem  = reinterpret_cast<half*>(smem_raw);
    half* dO_smem = Q_smem + BWD_BR * d_padded;

    // Thread → KV row mapping (same scheme as forward).
    const int my_kv_local  = tid % BWD_BC;
    const int my_kv_global = kv_start + my_kv_local;
    const int tpr = BWD_NUM_THREADS / BWD_BC;  // threads per row
    const int d_tid = tid / BWD_BC;
    const int d_per_thread = D_dim / tpr;
    const int d_off = d_tid * d_per_thread;

    // Load K, V rows for this KV block into registers.
    float k_reg[128], v_reg[128];
    for (int i = 0; i < d_per_thread; ++i) {
        int d_idx = d_off + i;
        if (my_kv_global < N && d_idx < D_dim) {
            k_reg[i] = __half2float(K_bh[my_kv_global * D_dim + d_idx]);
            v_reg[i] = __half2float(V_bh[my_kv_global * D_dim + d_idx]);
        } else {
            k_reg[i] = 0.0f;
            v_reg[i] = 0.0f;
        }
    }

    // Accumulators for dK, dV (in registers, written once at the end).
    float dk_acc[128], dv_acc[128];
    for (int i = 0; i < d_per_thread; ++i) {
        dk_acc[i] = 0.0f;
        dv_acc[i] = 0.0f;
    }

    // ── Inner loop over Q blocks ────────────────────────────────────
    const int num_q_blocks = cdiv(N, BWD_BR);
    int q_start_block = 0;
    if (causal) {
        // Only Q blocks where q_start >= kv_start matter.
        q_start_block = kv_start / BWD_BR;
    }

    for (int q_block = q_start_block; q_block < num_q_blocks; ++q_block) {
        const int q_start = q_block * BWD_BR;
        const int q_valid = min(BWD_BR, N - q_start);

        // Load Q and dO blocks into shared memory.
        smem_load_tile(Q_smem,  Q_bh  + q_start * D_dim,
                       BWD_BR, D_dim, tid, BWD_NUM_THREADS, q_valid);
        smem_load_tile(dO_smem, dO_bh + q_start * D_dim,
                       BWD_BR, D_dim, tid, BWD_NUM_THREADS, q_valid);
        __syncthreads();

        // For each Q row in this block, recompute S, P and accumulate grads.
        for (int qi = 0; qi < BWD_BR; ++qi) {
            const int q_global = q_start + qi;
            if (q_global >= N) break;

            // Causal: skip if this Q row is before all KV rows in this block.
            if (causal && q_global < kv_start) continue;

            // ── Recompute s = dot(Q_row, K_row) * scale ────────────
            float s_val;
            {
                float dot = 0.0f;
                for (int i = 0; i < d_per_thread; ++i) {
                    int d_idx = d_off + i;
                    dot += __half2float(Q_smem[qi * d_padded + d_idx]) * k_reg[i];
                }
                if (tpr == 2)
                    dot += __shfl_xor_sync(0xffffffff, dot, BWD_BC);
                s_val = dot * scale;
            }

            // Causal mask.
            if (causal && my_kv_global > q_global)
                s_val = -1e20f;
            if (my_kv_local >= kv_valid)
                s_val = -1e20f;

            // ── Recompute P = exp(S - L) ────────────────────────────
            float lse = L_bh[q_global];
            float p_val = __expf(s_val - lse);

            // ── dP = dot(dO_row, V_row) ─────────────────────────────
            float dp_val;
            {
                float dot = 0.0f;
                for (int i = 0; i < d_per_thread; ++i) {
                    int d_idx = d_off + i;
                    dot += __half2float(dO_smem[qi * d_padded + d_idx]) * v_reg[i];
                }
                if (tpr == 2)
                    dot += __shfl_xor_sync(0xffffffff, dot, BWD_BC);
                dp_val = dot;
            }

            // ── dS = P * (dP - D_i) ────────────────────────────────
            float d_i = D_bh[q_global];
            float ds_val = p_val * (dp_val - d_i);

            // ── Accumulate dV += p * dO_row  (outer product contrib) ─
            for (int i = 0; i < d_per_thread; ++i) {
                int d_idx = d_off + i;
                dv_acc[i] += p_val * __half2float(dO_smem[qi * d_padded + d_idx]);
            }

            // ── Accumulate dK += ds * Q_row * scale ─────────────────
            for (int i = 0; i < d_per_thread; ++i) {
                int d_idx = d_off + i;
                dk_acc[i] += ds_val * __half2float(Q_smem[qi * d_padded + d_idx]) * scale;
            }

            // ── dQ[q_global] += ds * K_row * scale  (atomicAdd) ─────
            // Only one partner thread does the atomic to avoid doubles.
            for (int i = 0; i < d_per_thread; ++i) {
                int d_idx = d_off + i;
                if (d_idx < D_dim && my_kv_local < kv_valid) {
                    atomicAdd(&dQ_bh[q_global * D_dim + d_idx],
                              ds_val * k_reg[i] * scale);
                }
            }
        }

        __syncthreads();
    }

    // ── Write dK, dV back to HBM ───────────────────────────────────
    if (my_kv_global < N) {
        for (int i = 0; i < d_per_thread; ++i) {
            int d_idx = d_off + i;
            if (d_idx < D_dim) {
                dK_bh[my_kv_global * D_dim + d_idx] = __float2half(dk_acc[i]);
                dV_bh[my_kv_global * D_dim + d_idx] = __float2half(dv_acc[i]);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Host launcher
// ════════════════════════════════════════════════════════════════════════

void flash_attn_bwd(
    const half*  Q,
    const half*  K,
    const half*  V,
    const half*  O,
    const half*  dO,
    const float* L,       // logsumexp from forward
    half*        dQ,      // output
    half*        dK,      // output
    half*        dV,      // output
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream)
{
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    // --- Step 1: compute D_i = rowsum(dO * O) ---
    float* d_D_vec = nullptr;
    CUDA_CHECK(cudaMalloc(&d_D_vec, (size_t)BH * N * sizeof(float)));
    {
        int threads = 256;
        dim3 grid(cdiv(N, threads), BH);
        compute_D_kernel<<<grid, threads, 0, stream>>>(dO, O, d_D_vec, N, D);
    }

    // --- Step 2: allocate float dQ accumulator (for atomicAdd) ---
    float* dQ_f = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_f, (size_t)BH * N * D * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_f, 0, (size_t)BH * N * D * sizeof(float), stream));

    // --- Step 3: main backward kernel ---
    {
        const int d_padded = padded_stride(D);
        int smem_bytes = 2 * BWD_BR * d_padded * sizeof(half);

        set_smem_limit(flash_attn_bwd_kernel, smem_bytes);

        dim3 grid(cdiv(N, BWD_BC), BH);
        dim3 block(BWD_NUM_THREADS);
        flash_attn_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, dO, L, d_D_vec,
            dQ_f, dK, dV,
            N, D, scale, causal);
        CUDA_CHECK_LAST();
    }

    // --- Step 4: convert float dQ back to half ---
    {
        // Simple element-wise kernel.
        size_t total = (size_t)BH * N * D;
        int threads = 256;
        int blocks = cdiv(static_cast<int>(total), threads);

        // Lambda-style kernel via __global__ below.
        // We'll use a tiny standalone kernel.
        // (Defined as a lambda would require --extended-lambda which we have.)
        auto convert = [] __global__ (const float* __restrict__ src,
                                       half* __restrict__ dst,
                                       size_t n) {
            size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n)
                dst[idx] = __float2half(src[idx]);
        };
        convert<<<blocks, threads, 0, stream>>>(dQ_f, dQ, total);
        CUDA_CHECK_LAST();
    }

    CUDA_CHECK(cudaFree(dQ_f));
    CUDA_CHECK(cudaFree(d_D_vec));
}
