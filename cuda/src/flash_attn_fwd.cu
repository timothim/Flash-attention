/**
 * flash_attn_fwd.cu — Flash Attention 2 forward pass.
 *
 * Key ideas:
 *   1. Tile the Q/KV dimensions into blocks of size Br × Bc.
 *   2. Q block stays in registers for the entire KV loop.
 *   3. K, V blocks are streamed from HBM into shared memory one at a time.
 *   4. Online softmax (Milakov & Gimelshein 2018) avoids materialising N×N.
 *   5. Two GEMMs per iteration: S = Q @ K^T (score), O += P @ V (accumulate).
 *
 * This file implements a FMA-based GEMM (Option A — compatible SM >= 6.0).
 * Each thread is responsible for a small sub-tile of the output, iterating
 * over the reduction (head_dim / Bc) dimension.
 *
 * Memory layout: all tensors are [B*H, N, D] row-major (batch and heads are
 * fused into the leading dimension).
 */

#include "cuda_utils.cuh"
#include "smem_utils.cuh"
#include "online_softmax.cuh"

// ════════════════════════════════════════════════════════════════════════
// Template parameters and thread-block mapping
//
//   Br = number of Q rows per block   (e.g. 64)
//   Bc = number of KV rows per block  (e.g. 64)
//   D  = head dimension               (e.g. 64 or 128)
//
//   Block shape : (NUM_THREADS)  — 1-D block
//   Grid        : (cdiv(N, Br),  B*H)
//
//   Within a block, threads cooperatively compute the two GEMMs and the
//   online-softmax update.  The mapping is:
//
//     Each thread "owns" TQ rows of the Q block (Br / TQ_PER_THREAD rows)
//     and iterates over all Bc columns for the S score tile, then over
//     all D columns for the O accumulator.
//
//   TQ = Br / NUM_THREADS_PER_ROW  — rows per thread
//   We use NUM_THREADS = 128 by default.
// ════════════════════════════════════════════════════════════════════════

// Fixed configuration — a good default for d=64 on A100.
// For d=128 you'd want Br=32, Bc=64 to fit in smem.
static constexpr int FWD_BR = 64;
static constexpr int FWD_BC = 64;
static constexpr int FWD_NUM_THREADS = 128;

// Each thread owns ROWS_PER_THREAD rows of the Br-row Q tile.
static constexpr int ROWS_PER_THREAD = FWD_BR / FWD_NUM_THREADS;
// When Br < NUM_THREADS, each thread handles at most 1 row but some
// threads are idle for row-parallel work; we clamp to 1.
static constexpr int RPT = (ROWS_PER_THREAD > 0) ? ROWS_PER_THREAD : 1;

// Alternatively use a 2-D thread mapping for GEMM. Here we use a simple
// scheme: threads 0..Br-1 each own exactly 1 row when Br <= NUM_THREADS.
// For Br=64, NUM_THREADS=128 → 2 threads per row, each handling half of
// the inner-dimension work.  We reduce across the pair via shfl.

__global__ void flash_attn_fwd_kernel(
    const half* __restrict__ Q,    // [B*H, N, D]
    const half* __restrict__ K,    // [B*H, N, D]
    const half* __restrict__ V,    // [B*H, N, D]
    half*       __restrict__ O,    // [B*H, N, D]
    float*      __restrict__ L,    // [B*H, N]   logsumexp
    const int N,
    const int D,
    const float scale,
    const bool causal)
{
    const int bh         = blockIdx.y;          // batch * head index
    const int block_q    = blockIdx.x;          // which Q block
    const int q_start    = block_q * FWD_BR;    // global row offset of Q block
    const int tid        = threadIdx.x;

    // Pointers into this (batch, head) slice.
    const half* Q_bh = Q + (size_t)bh * N * D;
    const half* K_bh = K + (size_t)bh * N * D;
    const half* V_bh = V + (size_t)bh * N * D;
    half*       O_bh = O + (size_t)bh * N * D;
    float*      L_bh = L + (size_t)bh * N;

    // ── Shared memory layout (dynamic) ──────────────────────────────
    // K_smem [Bc][D + PAD]   and   V_smem [Bc][D + PAD]
    extern __shared__ char smem_raw[];
    const int d_padded = padded_stride(D);
    half* K_smem = reinterpret_cast<half*>(smem_raw);
    half* V_smem = K_smem + FWD_BC * d_padded;
    // Small scratch for cross-warp reductions (if needed).
    float* reduce_smem = reinterpret_cast<float*>(V_smem + FWD_BC * d_padded);

    // ── Load Q block into registers ─────────────────────────────────
    // Each thread loads elements for the Q rows it owns.
    // We assign thread tid to Q row (tid % FWD_BR) — so with 128 threads
    // and Br=64, two threads share each row (they'll split the D-loop).
    const int my_q_row_local = tid % FWD_BR;            // 0..Br-1
    const int my_q_row_global = q_start + my_q_row_local;
    const int threads_per_row = FWD_NUM_THREADS / FWD_BR;  // 2 for 128/64
    const int d_tid = tid / FWD_BR;                        // 0 or 1

    // Each thread stores its chunk of the Q row in registers.
    // With threads_per_row = 2, each handles D/2 elements.
    const int d_per_thread = D / threads_per_row;
    const int d_offset = d_tid * d_per_thread;

    // Register arrays for Q row fragment, O accumulator, S scores.
    float q_reg[128];   // max D = 128
    float o_acc[128];   // max D = 128
    float s_row[64];    // max Bc = 64 — one row of the S_block

    // Load Q into registers (only the chunk this thread is responsible for).
    for (int i = 0; i < d_per_thread; ++i) {
        int d_idx = d_offset + i;
        if (my_q_row_global < N && d_idx < D)
            q_reg[i] = __half2float(Q_bh[my_q_row_global * D + d_idx]);
        else
            q_reg[i] = 0.0f;
    }

    // Initialise O accumulator to zero.
    for (int i = 0; i < d_per_thread; ++i)
        o_acc[i] = 0.0f;

    // Online softmax running state for this row.
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // ── Main KV loop ────────────────────────────────────────────────
    const int num_kv_blocks = cdiv(N, FWD_BC);
    int kv_end = num_kv_blocks;
    if (causal) {
        // With causal masking, we can skip KV blocks that are entirely
        // to the right of the Q block's last row.
        kv_end = cdiv(q_start + FWD_BR, FWD_BC);
        if (kv_end > num_kv_blocks) kv_end = num_kv_blocks;
    }

    for (int kv_block = 0; kv_block < kv_end; ++kv_block) {
        const int kv_start = kv_block * FWD_BC;
        const int kv_valid = min(FWD_BC, N - kv_start);  // valid rows in this KV block

        // ── Load K, V into shared memory ────────────────────────────
        smem_load_tile(K_smem, K_bh + kv_start * D,
                       FWD_BC, D, tid, FWD_NUM_THREADS, kv_valid);
        smem_load_tile(V_smem, V_bh + kv_start * D,
                       FWD_BC, D, tid, FWD_NUM_THREADS, kv_valid);
        __syncthreads();

        // ── GEMM-I:  s_row[j] = dot(Q_row, K_row[j]) × scale ──────
        // Each thread computes partial dots for its D chunk, then
        // reduces across the threads_per_row partners via shfl.
        for (int j = 0; j < FWD_BC; ++j) {
            float dot = 0.0f;
            for (int i = 0; i < d_per_thread; ++i) {
                int d_idx = d_offset + i;
                dot += q_reg[i] * __half2float(K_smem[j * d_padded + d_idx]);
            }
            // Reduce across partner threads sharing this Q row.
            if (threads_per_row == 2) {
                dot += __shfl_xor_sync(0xffffffff, dot, FWD_BR);
            }
            dot *= scale;

            // Causal mask.
            if (causal) {
                int global_j = kv_start + j;
                if (global_j > my_q_row_global)
                    dot = -1e20f;
            }

            // Mask padding.
            if (j >= kv_valid)
                dot = -1e20f;

            s_row[j] = dot;
        }

        // ── Online softmax update ───────────────────────────────────
        // Find row max over s_row.
        float m_block = -FLT_MAX;
        for (int j = 0; j < FWD_BC; ++j)
            m_block = fmaxf(m_block, s_row[j]);

        float m_new = fmaxf(m_i, m_block);
        float alpha = __expf(m_i - m_new);      // rescale for old accum
        // Compute exp(s - m_new) and row sum.
        float row_sum = 0.0f;
        for (int j = 0; j < FWD_BC; ++j) {
            s_row[j] = __expf(s_row[j] - m_new);
            row_sum += s_row[j];
        }

        // Update running stats.
        l_i = alpha * l_i + row_sum;
        m_i = m_new;

        // Rescale old O accumulator.
        for (int i = 0; i < d_per_thread; ++i)
            o_acc[i] *= alpha;

        // ── GEMM-II:  O_acc += P_row @ V_smem ──────────────────────
        // P_row = s_row (already contains exp(s - m_new)).
        // Each thread accumulates its D chunk of the output row.
        for (int j = 0; j < FWD_BC; ++j) {
            float p = s_row[j];
            for (int i = 0; i < d_per_thread; ++i) {
                int d_idx = d_offset + i;
                o_acc[i] += p * __half2float(V_smem[j * d_padded + d_idx]);
            }
        }

        __syncthreads();  // Before loading next KV block into smem.
    }

    // ── Final normalisation and write-back ──────────────────────────
    if (my_q_row_global < N) {
        float inv_l = 1.0f / (l_i + 1e-8f);
        for (int i = 0; i < d_per_thread; ++i) {
            int d_idx = d_offset + i;
            if (d_idx < D) {
                O_bh[my_q_row_global * D + d_idx] = __float2half(o_acc[i] * inv_l);
            }
        }
        // Only one of the partner threads writes L (avoid duplicate writes).
        if (d_tid == 0) {
            L_bh[my_q_row_global] = m_i + logf(l_i + 1e-8f);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Host launcher
// ════════════════════════════════════════════════════════════════════════

void flash_attn_fwd(
    const half* Q,
    const half* K,
    const half* V,
    half*       O,
    float*      L,
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream)
{
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    // Shared memory size.
    const int d_padded = padded_stride(D);
    int smem_bytes = 2 * FWD_BC * d_padded * sizeof(half)
                   + 32 * sizeof(float);  // reduce scratch

    // May need to opt into larger smem.
    set_smem_limit(flash_attn_fwd_kernel, smem_bytes);

    dim3 grid(cdiv(N, FWD_BR), BH);
    dim3 block(FWD_NUM_THREADS);

    flash_attn_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, L, N, D, scale, causal);

    CUDA_CHECK_LAST();
}
