"""
Unified benchmark script for all attention implementations.

Implementations tested:
  1. Naive (CUDA)           — materialises N×N
  2. Flash Attention (CUDA) — our custom kernel
  3. PyTorch SDPA           — torch.nn.functional.scaled_dot_product_attention
  4. Triton                 — our Triton kernel
  5. Tri Dao flash-attn     — pip package (if installed)

Results are saved as JSON in benchmarks/results/ and can be plotted with
plot_results.py.
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn.functional as F

# Add project dirs to path.
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "triton"))

# ── Import implementations ───────────────────────────────────────────

try:
    import flash_attn_cuda
    HAS_CUDA = True
except ImportError:
    print("WARNING: flash_attn_cuda not found. Run `make install` first.")
    HAS_CUDA = False

try:
    from flash_attn_triton import flash_attn_triton_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_func as dao_flash_attn_func
    HAS_DAO = True
except ImportError:
    HAS_DAO = False


# ── Benchmark harness ────────────────────────────────────────────────

def bench_fn(fn, warmup=10, iters=100):
    """Benchmark a CUDA function. Returns average latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compute_metrics(B, H, N, D, latency_ms):
    """Compute TFLOPS and bandwidth utilisation."""
    flops_fwd = 4.0 * B * H * N * N * D
    tflops = flops_fwd / (latency_ms * 1e-3) / 1e12

    # Bytes accessed (approximate): read Q,K,V,O + write O + L.
    bytes_accessed = B * H * (4 * N * D + N) * 2  # FP16
    bw_gb_s = bytes_accessed / (latency_ms * 1e-3) / 1e9

    return tflops, bw_gb_s


# ── Main benchmark loop ─────────────────────────────────────────────

def run_benchmarks():
    device = "cuda"
    results = []

    # Fixed params.
    B = 4
    H = 16
    seq_lens = [256, 512, 1024, 2048, 4096]
    head_dims = [64, 128]
    causal_modes = [False, True]

    for D in head_dims:
        for causal in causal_modes:
            for N in seq_lens:
                print(f"\n--- D={D}  N={N}  causal={causal} ---")

                Q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
                K = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
                V = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

                config = {"B": B, "H": H, "N": N, "D": D, "causal": causal}

                # 1. Naive (skip for large N — would OOM).
                if HAS_CUDA and N <= 2048:
                    try:
                        ms = bench_fn(lambda: flash_attn_cuda.naive_fwd(Q, K, V, causal))
                        tflops, bw = compute_metrics(B, H, N, D, ms)
                        peak_mem = torch.cuda.max_memory_allocated() / 1e6
                        torch.cuda.reset_peak_memory_stats()
                        entry = {**config, "impl": "naive", "latency_ms": round(ms, 4),
                                 "tflops": round(tflops, 2), "bw_gb_s": round(bw, 1),
                                 "peak_mem_mb": round(peak_mem, 1)}
                        results.append(entry)
                        print(f"  Naive:    {ms:.3f} ms  {tflops:.2f} TFLOPS")
                    except RuntimeError as e:
                        print(f"  Naive:    SKIPPED ({e})")

                # 2. Flash CUDA.
                if HAS_CUDA:
                    torch.cuda.reset_peak_memory_stats()
                    ms = bench_fn(lambda: flash_attn_cuda.flash_fwd(Q, K, V, causal))
                    tflops, bw = compute_metrics(B, H, N, D, ms)
                    peak_mem = torch.cuda.max_memory_allocated() / 1e6
                    torch.cuda.reset_peak_memory_stats()
                    entry = {**config, "impl": "flash_cuda", "latency_ms": round(ms, 4),
                             "tflops": round(tflops, 2), "bw_gb_s": round(bw, 1),
                             "peak_mem_mb": round(peak_mem, 1)}
                    results.append(entry)
                    print(f"  Flash CUDA: {ms:.3f} ms  {tflops:.2f} TFLOPS")

                # 3. PyTorch SDPA.
                torch.cuda.reset_peak_memory_stats()
                ms = bench_fn(lambda: F.scaled_dot_product_attention(
                    Q, K, V, is_causal=causal))
                tflops, bw = compute_metrics(B, H, N, D, ms)
                peak_mem = torch.cuda.max_memory_allocated() / 1e6
                torch.cuda.reset_peak_memory_stats()
                entry = {**config, "impl": "pytorch_sdpa", "latency_ms": round(ms, 4),
                         "tflops": round(tflops, 2), "bw_gb_s": round(bw, 1),
                         "peak_mem_mb": round(peak_mem, 1)}
                results.append(entry)
                print(f"  PyTorch SDPA: {ms:.3f} ms  {tflops:.2f} TFLOPS")

                # 4. Triton.
                if HAS_TRITON:
                    torch.cuda.reset_peak_memory_stats()
                    ms = bench_fn(lambda: flash_attn_triton_fwd(Q, K, V, causal))
                    tflops, bw = compute_metrics(B, H, N, D, ms)
                    peak_mem = torch.cuda.max_memory_allocated() / 1e6
                    torch.cuda.reset_peak_memory_stats()
                    entry = {**config, "impl": "triton", "latency_ms": round(ms, 4),
                             "tflops": round(tflops, 2), "bw_gb_s": round(bw, 1),
                             "peak_mem_mb": round(peak_mem, 1)}
                    results.append(entry)
                    print(f"  Triton:   {ms:.3f} ms  {tflops:.2f} TFLOPS")

                # 5. Tri Dao flash-attn.
                if HAS_DAO:
                    # Dao's API expects [B, N, H, D].
                    Q_dao = Q.transpose(1, 2).contiguous()
                    K_dao = K.transpose(1, 2).contiguous()
                    V_dao = V.transpose(1, 2).contiguous()
                    torch.cuda.reset_peak_memory_stats()
                    ms = bench_fn(lambda: dao_flash_attn_func(
                        Q_dao, K_dao, V_dao, causal=causal))
                    tflops, bw = compute_metrics(B, H, N, D, ms)
                    peak_mem = torch.cuda.max_memory_allocated() / 1e6
                    torch.cuda.reset_peak_memory_stats()
                    entry = {**config, "impl": "dao_flash", "latency_ms": round(ms, 4),
                             "tflops": round(tflops, 2), "bw_gb_s": round(bw, 1),
                             "peak_mem_mb": round(peak_mem, 1)}
                    results.append(entry)
                    print(f"  Dao Flash: {ms:.3f} ms  {tflops:.2f} TFLOPS")

    # Save results.
    out_dir = os.path.join(PROJ_ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_benchmarks()
