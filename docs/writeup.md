# Flash Attention 2: From Algorithm to CUDA Kernel

A technical writeup on implementing Flash Attention 2 from scratch in CUDA C++ and Triton, covering the algorithm, memory hierarchy mapping, implementation details, and performance analysis.

## 1. Introduction — Why Flash Attention

Standard multi-head attention computes S = QK^T (an N×N matrix), applies softmax row-wise, then multiplies P×V. The arithmetic complexity is O(N²d) — manageable for modern GPUs. The problem is memory.

On an A100, the HBM bandwidth is ~2 TB/s and the FP16 Tensor Core peak is 312 TFLOPS. The crossover point on the roofline — where a kernel transitions from memory-bound to compute-bound — sits at roughly 156 FLOPs/byte. Standard attention's arithmetic intensity is far below this threshold: for each element of the N×N score matrix, we do O(d) FLOPs but must read/write from HBM, yielding an intensity of roughly O(d) FLOPs per O(1) bytes. With d=64 or 128, we're solidly in the memory-bound regime.

The N×N matrix itself is the culprit. For a sequence of length 4096 with batch 4 and 16 heads, materialising S in FP16 costs 4 × 16 × 4096² × 2 bytes = 2 GB — just for a single intermediate. This blows out the L2 cache (40 MB on A100) and forces repeated round-trips to HBM.

Flash Attention eliminates this bottleneck by never materialising the full N×N matrix. Instead, it processes attention in tiles that fit in the GPU's on-chip SRAM (shared memory + registers), computing the softmax incrementally via the online softmax algorithm. The result is O(N) memory instead of O(N²), and a significant reduction in HBM traffic that translates directly into wall-clock speedup — not because we do fewer FLOPs, but because we move fewer bytes.

## 2. The Algorithm

### Online Softmax

The key insight enabling tiled attention is that softmax can be computed incrementally. The standard softmax requires two passes over the full row: one to find the max (for numerical stability), another to compute exp and sum. With an N×N matrix, this means the entire row must be in memory simultaneously.

Milakov and Gimelshein (2018) showed that these running statistics can be maintained incrementally. We keep a running max `m` and a running sum-of-exponentials `l`. When a new block of scores arrives, we update:

```
m_new = max(m_old, max(new_block))
alpha = exp(m_old - m_new)          # rescale factor for old accumulator
l_new = alpha * l_old + sum(exp(new_block - m_new))
```

The crucial property is that the output accumulator O can also be rescaled: `O_new = alpha * O_old + P_block @ V_block`, where `P_block` uses the updated normalisation. At the end, a single division by `l` produces the correctly normalised output.

### Tiling Strategy

Flash Attention 2 organises the computation as a doubly-nested loop. The outer loop iterates over Q blocks (each of size Br rows), and the inner loop streams KV blocks (each of size Bc rows) from HBM into shared memory.

The asymmetry is intentional. Q stays pinned in registers for the entire inner loop — it's read once from HBM and reused across every KV block. K and V are loaded into shared memory one block at a time, used for two GEMMs (S = Q @ K^T and O += P @ V), then discarded. This means HBM reads for K and V are O(N/Bc) per Q block, and Q is read O(1) times total per block — a massive reduction vs the naive approach where every element of Q, K, V is read O(N) times.

### Backward Pass — Recomputation Trade-off

The backward pass needs P (the softmax output) to compute gradients. Storing P would cost O(N²) memory, defeating the purpose. Flash Attention instead saves only the logsumexp vector L (O(N) memory) and recomputes P on the fly: P_ij = exp(S_ij - L_i).

The backward loop is inverted: the outer loop iterates over KV blocks (so dK and dV accumulate in registers), and the inner loop iterates over Q blocks. dQ is accumulated via atomicAdd since multiple KV blocks contribute to the same Q row's gradient.

This recomputation trade-off — spending extra FLOPs to avoid storing O(N²) data — is profitable precisely because standard attention is memory-bound. The extra compute is "free" in the sense that the GPU's arithmetic units were underutilised anyway.

## 3. CUDA Implementation

### Memory Layout and Thread Mapping

All tensors use [B*H, N, D] row-major layout with batch and head dimensions fused. This makes the per-block pointer arithmetic trivial: each thread block handles one (batch, head, Q-block) triplet.

With `NUM_THREADS=128` and `Br=64`, we assign two threads per Q row. Each thread handles D/2 elements of the head dimension. The two partners compute partial dot products independently and reduce via `__shfl_xor_sync` — a single warp shuffle instruction that costs essentially nothing compared to a shared memory reduction.

### Shared Memory and Bank Conflicts

Shared memory on NVIDIA GPUs is divided into 32 banks, each 4 bytes wide. When D=64, a row of K in shared memory (64 half values = 128 bytes) spans exactly 32 banks. This means column-wise access patterns — which occur during the K^T multiplication — produce 32-way bank conflicts, serialising the entire warp's memory access.

The fix is padding: we allocate `K_smem[Bc][D + 8]` instead of `K_smem[Bc][D]`. The extra 8 halfs (16 bytes) shift each row's bank alignment, breaking the conflict pattern. This is a well-known technique, but getting the padding constant right requires understanding the exact bank layout — padding by 4 halfs (8 bytes = 2 banks) would be insufficient for D=64.

With Br=Bc=64 and D=64, the shared memory footprint per block is:

```
K_smem: 64 × (64 + 8) × 2 bytes = 9,216 bytes
V_smem: 64 × (64 + 8) × 2 bytes = 9,216 bytes
Total: ~18.4 KB per block
```

This fits comfortably within the A100's 164 KB shared memory per SM, allowing multiple blocks to co-reside (good occupancy). For D=128, the footprint doubles to ~36.8 KB, which still permits at least 2 blocks per SM.

### Warp-Level Reductions

The online softmax requires a row-wise max and sum after each S-block computation. Since each row of S is distributed across threads (possibly across warps), we need efficient reductions.

Within a warp, butterfly reduction via `__shfl_xor_sync` computes the max/sum in 5 steps (log2(32) = 5 shuffles). Each shuffle exchanges data between thread pairs at increasing distances (16, 8, 4, 2, 1), converging to the global value in all threads simultaneously.

When a row spans multiple warps, we fall back to a shared memory reduction: each warp writes its partial result, a sync barrier ensures visibility, then warp 0 reads and reduces the partial results. This two-level scheme (intra-warp shfl + inter-warp smem) is standard practice in high-performance CUDA kernels.

### The GEMMs

Both GEMMs (S = Q @ K^T and O += P @ V) are implemented as straightforward FMA loops. Each thread iterates over its assigned D/2 elements, accumulating in float32 registers. This is the "Option A" approach — compatible with all GPUs from SM 6.0 (Pascal) onward, without requiring Tensor Core hardware.

The inner loop for GEMM-I looks like:

```cpp
for (int j = 0; j < Bc; ++j) {
    float dot = 0.0f;
    for (int i = 0; i < d_per_thread; ++i)
        dot += q_reg[i] * __half2float(K_smem[j * d_padded + d_idx]);
    // reduce across partner thread, apply scale, store in s_row[j]
}
```

This is far from optimal — a Tensor Core WMMA implementation would deliver 8-16x higher throughput on the GEMM alone. But the goal here is correctness and clarity first. The FMA path makes the data flow explicit and serves as a baseline for understanding where Tensor Cores would plug in.

### Debugging War Stories

The most insidious bug was in the backward pass logsumexp indexing. The forward computes `L[i] = m_i + log(l_i)`, and the backward recomputes `P = exp(S - L)`. If L is computed with a small epsilon (`log(l_i + 1e-8)`) but the backward assumes exact logsumexp, the gradients develop a systematic bias that grows with sequence length. The fix was ensuring the same epsilon appears in both forward and backward — a reminder that numerical consistency between passes matters more than absolute precision in either one.

A second class of bugs came from boundary handling. When N is not a multiple of Br or Bc, the last tile is partial. Failing to mask padding positions in the S-block score computation doesn't just produce wrong values — it can produce NaN in the softmax (since uninitialised smem contains garbage, and exp(garbage) overflows). The solution is writing `-inf` to masked positions before the softmax, so they contribute zero probability.

## 4. Triton Implementation

### What Triton Abstracts

Triton eliminates the need to manually manage shared memory allocation, padding, tile loads, thread-to-data mapping, and register allocation. A `tl.load` with a mask handles boundary conditions automatically. `tl.dot` compiles to efficient GEMM instructions (including Tensor Core paths when available). The programmer thinks in terms of 2D blocks, not individual threads.

The entire forward kernel is roughly 60 lines of Python, compared to ~200 lines of CUDA. The backward is similarly compact.

### What We Lose

The flip side of Triton's abstractions is reduced control. We can't tune bank-conflict patterns (Triton's compiler makes its own smem layout decisions), can't use explicit warp shuffles, and can't control register allocation. On some configurations, Triton's auto-generated PTX shows suboptimal register spilling that a hand-tuned CUDA kernel avoids.

### Autotuning

Triton's autotune decorator tests four configurations varying BLOCK_Q (64/128), BLOCK_KV (64/128), num_warps (4/8), and num_stages (2/3). The autotuner runs each configuration on the actual input shape and selects the fastest. In practice, the 64×64/4-warp configuration tends to win for D=64, while 128×64/4-warp wins for D=128 — consistent with the larger smem footprint of D=128 tiles.

## 5. Results

The benchmark suite tests five implementations across sequence lengths from 256 to 8192, for head dimensions 64 and 128, with and without causal masking.

Key observations (numbers are GPU-dependent and should be verified on the target hardware by running `make bench`):

**Memory scaling**: The naive baseline shows quadratic memory growth — at N=4096 it allocates several GB for the score matrix alone. Flash Attention (both CUDA and Triton) maintains near-constant memory overhead regardless of sequence length, confirming the O(N) memory guarantee.

**Throughput**: PyTorch SDPA (which dispatches to an optimised Flash Attention backend internally) is the fastest, since it uses Tensor Cores and has been tuned by NVIDIA/Meta engineers. Our Triton kernel typically achieves good throughput thanks to the compiler's automatic Tensor Core utilisation. Our CUDA kernel, using FMA-only GEMMs, is compute-limited on the GEMM steps — the gap to SDPA reflects the Tensor Core advantage.

**Causal masking**: Causal mode provides a ~2x throughput improvement at long sequences because roughly half the KV blocks can be skipped entirely (they're entirely masked out). This early-exit optimisation is present in all Flash Attention implementations.

**CUDA vs Triton**: Triton benefits from automatic Tensor Core lowering that our FMA-based CUDA kernel lacks. For an apples-to-apples comparison, a WMMA-enabled CUDA kernel would close this gap.

## 6. Limitations and Extensions

**Multi-Head vs Grouped-Query Attention**: This implementation assumes standard MHA where every head has its own Q, K, V. GQA (used in Llama 2/3 and others) shares K/V across groups of query heads — supporting this requires minor changes to the indexing logic.

**Variable-Length Sequences**: We assume all sequences in the batch have the same length. Production systems use padding or packed sequences with a cumulative-length index, which requires a different block scheduling strategy.

**Paged KV Cache**: For autoregressive inference, the KV cache grows token by token. PagedAttention (vLLM) manages this via virtual memory-style paging. Our kernel doesn't handle non-contiguous KV blocks.

**Flash Attention 3**: On Hopper (H100), Flash Attention 3 leverages hardware features not available on Ampere: TMA (Tensor Memory Accelerator) for asynchronous global-to-shared copies, WGMMA (Warpgroup Matrix Multiply-Accumulate) for larger and more efficient Tensor Core operations, and asynchronous pipelining that overlaps GEMM compute with the next tile's memory load. These features are architecturally significant and represent the next frontier for hand-optimised attention kernels.
