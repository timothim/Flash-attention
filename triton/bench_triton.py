"""
Standalone benchmark for the Triton Flash Attention implementation.
"""

import torch
import math
import json
import time
from flash_attn_triton import flash_attn_triton_fwd


def bench_triton(B=4, H=16, D=64, seq_lens=None, causal=False,
                 warmup=10, iters=100):
    if seq_lens is None:
        seq_lens = [256, 512, 1024, 2048, 4096]

    results = []
    for N in seq_lens:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        # Warmup.
        for _ in range(warmup):
            flash_attn_triton_fwd(Q, K, V, causal=causal)
        torch.cuda.synchronize()

        # Timed runs.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            flash_attn_triton_fwd(Q, K, V, causal=causal)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters

        flops = 4 * B * H * N * N * D
        tflops = flops / (ms * 1e-3) / 1e12
        results.append({
            "impl": "triton",
            "B": B, "H": H, "N": N, "D": D,
            "causal": causal,
            "latency_ms": round(ms, 4),
            "tflops": round(tflops, 2),
        })
        print(f"N={N:5d}  latency={ms:.3f} ms  TFLOPS={tflops:.2f}")

    return results


if __name__ == "__main__":
    print("=== Triton Flash Attention Benchmark ===\n")
    for D in [64, 128]:
        print(f"\n--- head_dim={D} ---")
        res = bench_triton(D=D)
    print("\nDone.")
