# Flash Attention 2 — From-Scratch Implementation

A ground-up implementation of [Flash Attention 2](https://arxiv.org/abs/2307.09288) (Dao, 2023) in **CUDA C++** and **Triton**, with no reliance on cuBLAS, cuDNN, or CUTLASS for the matrix multiplications. Every GEMM is hand-written using FMA loops at the thread level.

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-green.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project implements a complete Flash Attention 2 pipeline from scratch:

- **Forward pass**: Tiled Q×K^T and P×V GEMMs with online softmax (Milakov & Gimelshein, 2018), avoiding materialisation of the N×N attention matrix
- **Backward pass**: Recomputes S and P from the saved logsumexp vector L instead of storing the O(N²) attention matrix
- **Triton version**: Mirrors the same algorithm using `tl.dot`, `tl.load/store`, and Triton's autotuning over block sizes and warp counts
- **PyTorch integration**: C++ extension with full autograd support

This is the first module of a four-part **Deep Systems Portfolio** (GPU → OS → Framework → Model) designed to demonstrate hardware-level understanding of modern ML infrastructure.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT TENSORS                            │
│            Q, K, V : [B, H, N, D] — FP16 CUDA tensors            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      TILED OUTER LOOP                            │
│         Q blocks (Br rows) stay in registers                     │
│         K, V blocks (Bc rows) stream through shared memory       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│       GEMM-I            │   │       GEMM-II           │
│   S = Q @ K^T × scale   │   │   O += softmax(S) @ V   │
│   (FMA dot products)    │   │   (FMA accumulation)    │
└─────────────────────────┘   └─────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     ONLINE SOFTMAX UPDATE                        │
│   m_new = max(m_old, max(S_block))                               │
│   l_new = exp(m_old - m_new) × l_old + sum(exp(S - m_new))       │
│   O_new = α × O_old + P_block @ V_block                          │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
│   O : [B, H, N, D] — Normalized attention output                 │
│   L : [B, H, N]    — Logsumexp (saved for backward)              │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- NVIDIA GPU (SM 7.0+ / Volta or newer)
- CUDA 12.0+
- PyTorch 2.0+ with CUDA support
- Triton 2.1+ (for the Triton kernel)
- Python 3.10+

### Installation

```bash
# Clone repository
git clone https://github.com/timothim/Flash-attention.git
cd Flash-attention

# Build and install the CUDA PyTorch extension
make install
```

### Testing

```bash
# Run CUDA correctness tests
make test

# Run Triton tests
make test-triton

# Run all tests
make test-all
```

### Benchmarking

```bash
# Full benchmark suite (all implementations)
make bench

# Generate plots from results
make plots
```

## Project Structure

```
01-flash-attention/
├── cuda/
│   ├── include/
│   │   ├── flash_attn.cuh          # Public kernel API
│   │   ├── online_softmax.cuh      # Warp reductions + online softmax state
│   │   ├── smem_utils.cuh          # Padded tile loads (anti bank-conflict)
│   │   └── cuda_utils.cuh          # CUDA_CHECK, timing, cdiv
│   │
│   ├── src/
│   │   ├── flash_attn_fwd.cu       # Flash Attention 2 forward kernel
│   │   ├── flash_attn_bwd.cu       # Flash Attention 2 backward kernel
│   │   ├── naive_attention.cu      # Baseline (materialises N×N)
│   │   └── torch_bindings.cpp      # PyTorch C++ extension
│   │
│   ├── tests/
│   │   ├── test_correctness.py     # Forward + backward vs PyTorch reference
│   │   └── test_edge_cases.py      # Non-aligned seq_len, batch=1, etc.
│   │
│   ├── CMakeLists.txt              # CMake build (multi-arch)
│   └── setup.py                    # pip install -e .
│
├── triton/
│   ├── flash_attn_triton.py        # Triton forward + backward + autograd
│   ├── test_triton.py              # Correctness tests
│   └── bench_triton.py             # Standalone Triton benchmark
│
├── benchmarks/
│   ├── bench_all.py                # Unified benchmark (all implementations)
│   ├── plot_results.py             # Matplotlib plot generation
│   └── results/                    # JSON results + PNG plots
│
├── Makefile                        # make install / test / bench / plots
├── LICENSE                         # MIT License
└── README.md
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FMA-based GEMM** | Compatible with SM 6.0+, prioritizes clarity over peak throughput. Tensor Core (WMMA) path is a natural extension. |
| **Thread mapping** | With `NUM_THREADS=128` and `Br=64`, two threads share each Q row, splitting the head_dim work and reducing via `__shfl_xor_sync`. |
| **Shared memory padding** | K and V tiles use `+8` half padding per row to break 32-bank alignment that causes guaranteed conflicts when `D` is a multiple of 64. |
| **Backward atomicAdd** | Outer loop iterates over KV blocks (keeping dK/dV in registers), so multiple blocks contribute to the same dQ rows via `atomicAdd`. |

## Implementations Benchmarked

| Implementation | Description |
|----------------|-------------|
| **Naive** | Materialises full N×N score matrix. O(N²) memory. |
| **Flash CUDA (ours)** | Hand-written CUDA kernel, FMA GEMM, online softmax. |
| **PyTorch SDPA** | `torch.nn.functional.scaled_dot_product_attention` |
| **Triton (ours)** | Triton kernel with autotuning. |
| **Tri Dao flash-attn** | Reference implementation (if installed). |

## Mathematical Background

### Online Softmax (Milakov & Gimelshein, 2018)

Standard softmax requires two passes over the full row. Online softmax maintains running statistics incrementally:

```
m_new = max(m_old, max(new_block))
α = exp(m_old - m_new)                    # rescale factor
l_new = α × l_old + Σ exp(new_block - m_new)
O_new = α × O_old + P_block @ V_block
```

### Memory Complexity

| Algorithm | Memory | HBM Reads |
|-----------|--------|-----------|
| Standard Attention | O(N²) | O(N² × d) |
| Flash Attention | O(N) | O(N × d × N/Bc) |

The key insight: Q stays in registers, K/V stream through shared memory once per Q block.

## Numerical Tolerances

| Pass | Precision | Tolerance |
|------|-----------|-----------|
| Forward FP16 | vs float32 reference | < 2% relative error |
| Backward FP16 | vs float32 reference | < 10% relative error |

*FP16 backward is inherently less stable due to gradient accumulation.*

## Development

```bash
# Build and install
make install

# Run tests
make test

# Run benchmarks
make bench

# Clean build artifacts
make clean
```

## References

| Topic | Paper |
|-------|-------|
| Flash Attention | Dao et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135). NeurIPS 2022 |
| Flash Attention 2 | Dao (2023). [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.09288) |
| Online Softmax | Milakov & Gimelshein (2018). [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Tri Dao](https://github.com/Dao-AILab/flash-attention) — Original Flash Attention implementation
- [OpenAI Triton](https://github.com/openai/triton) — GPU programming language
- [PyTorch](https://pytorch.org/) — Deep learning framework
