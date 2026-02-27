# Flash Attention 2 — From-Scratch Implementation

A ground-up implementation of [Flash Attention 2](https://arxiv.org/abs/2307.09288) (Dao, 2023) in **CUDA C++** and **Triton**, with no reliance on cuBLAS, cuDNN, or CUTLASS for the matrix multiplications. Every GEMM is hand-written using FMA loops at the thread level.

This project is the first module of a four-part **Deep Systems Portfolio** (GPU → OS → Framework → Model) designed to demonstrate hardware-level understanding of modern ML infrastructure.

## What This Implements

**Forward pass**: Tiled Q×K^T and P×V GEMMs with online softmax (Milakov & Gimelshein, 2018), avoiding materialisation of the N×N attention matrix. Q tiles stay in registers; K/V tiles are streamed through shared memory.

**Backward pass**: Recomputes S and P from the saved logsumexp vector L instead of storing the O(N²) attention matrix. Outer loop on KV blocks with dK/dV accumulated in registers; inner loop on Q blocks with dQ accumulated via atomicAdd.

**Triton version**: Mirrors the same algorithm using `tl.dot`, `tl.load/store`, and Triton's autotuning over block sizes and warp counts.

## Project Structure

```
01-flash-attention/
├── cuda/
│   ├── CMakeLists.txt              # CMake build (multi-arch)
│   ├── include/
│   │   ├── flash_attn.cuh          # Public kernel API
│   │   ├── online_softmax.cuh      # Warp reductions + online softmax state
│   │   ├── smem_utils.cuh          # Padded tile loads (anti bank-conflict)
│   │   └── cuda_utils.cuh          # CUDA_CHECK, timing, cdiv
│   ├── src/
│   │   ├── flash_attn_fwd.cu       # Flash Attention 2 forward kernel
│   │   ├── flash_attn_bwd.cu       # Flash Attention 2 backward kernel
│   │   ├── naive_attention.cu      # Baseline (materialises N×N)
│   │   └── torch_bindings.cpp      # PyTorch C++ extension
│   ├── tests/
│   │   ├── test_correctness.py     # Forward + backward vs PyTorch reference
│   │   └── test_edge_cases.py      # Non-aligned seq_len, batch=1, etc.
│   └── setup.py                    # pip install -e .
├── triton/
│   ├── flash_attn_triton.py        # Triton forward + backward + autograd
│   ├── test_triton.py              # Correctness tests
│   └── bench_triton.py             # Standalone Triton benchmark
├── benchmarks/
│   ├── bench_all.py                # Unified benchmark (all implementations)
│   ├── plot_results.py             # Matplotlib plot generation
│   └── results/                    # JSON results + PNG plots
├── docs/
│   └── writeup.md                  # Technical writeup (~2500 words)
└── Makefile                        # make install / test / bench / plots
```

## Quick Start

### Prerequisites

- NVIDIA GPU (SM 7.0+ / Volta or newer)
- CUDA 12.0+
- PyTorch 2.0+ with CUDA support
- Triton 2.1+ (for the Triton kernel)
- Python 3.10+

### Build & Test

```bash
# Build and install the CUDA PyTorch extension
make install

# Run correctness tests
make test

# Run Triton tests
make test-triton

# Run all tests
make test-all
```

### Benchmark

```bash
# Full benchmark suite (all implementations)
make bench

# Generate plots from results
make plots
```

## Design Decisions

**FMA-based GEMM (no Tensor Cores by default)**: The GEMM is implemented as a manual FMA dot-product loop, compatible with SM 6.0+. This was chosen for portability and clarity. A Tensor Core (WMMA) path is a natural extension.

**Thread mapping**: With `NUM_THREADS=128` and `Br=64`, two threads share each Q row, splitting the head_dim work and reducing via `__shfl_xor_sync`. This keeps register pressure manageable while maintaining good occupancy.

**Shared memory padding**: K and V tiles in shared memory use `+8` half padding per row to break the 32-bank alignment that causes guaranteed conflicts when `D` is a multiple of 64.

**Backward atomicAdd for dQ**: Since the backward outer loop iterates over KV blocks (to keep dK/dV in registers), multiple blocks contribute to the same dQ rows. We accumulate in float32 via `atomicAdd` and convert to FP16 at the end.

## Implementations Benchmarked

| Implementation | Description |
|---|---|
| **Naive** | Materialises full N×N score matrix. O(N²) memory. |
| **Flash CUDA (ours)** | Hand-written CUDA kernel, FMA GEMM, online softmax. |
| **PyTorch SDPA** | `torch.nn.functional.scaled_dot_product_attention` |
| **Triton (ours)** | Triton kernel with autotuning. |
| **Tri Dao flash-attn** | Reference implementation (if installed). |

## Numerical Tolerances

- Forward FP16: relative error < 2% vs float32 reference
- Backward FP16: relative error < 10% vs float32 reference (expected — FP16 backward is inherently less stable due to gradient accumulation)

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
- Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv 1805.02867, 2018

## License

MIT
