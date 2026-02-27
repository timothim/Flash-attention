"""
Edge-case tests for Flash Attention kernels.

Covers non-standard configurations: seq_len not a multiple of block size,
batch=1, small head_dim, single-head, etc.
"""

import pytest
import torch
import math

import flash_attn_cuda


def ref_attention(Q, K, V, causal=False):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V.float())


def check_fwd(B, H, N, D, causal, tol=2e-2):
    torch.manual_seed(123)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    O_ref = ref_attention(Q, K, V, causal=causal).half()
    O_flash, _ = flash_attn_cuda.flash_fwd(Q, K, V, causal=causal)

    max_err = (O_flash.float() - O_ref.float()).abs().max().item()
    ref_scale = O_ref.float().abs().max().item() + 1e-5
    rel_err = max_err / ref_scale
    print(f"  B={B} H={H} N={N} D={D} causal={causal} | "
          f"max_err={max_err:.4e}  rel_err={rel_err:.4e}")
    assert rel_err < tol, f"Relative error too high: {rel_err}"


# ── Seq_len NOT a multiple of block size (64) ─────────────────────────

@pytest.mark.parametrize("N", [1, 7, 33, 63, 65, 100, 127, 129, 200])
def test_non_aligned_seqlen(N):
    check_fwd(B=1, H=2, N=N, D=64, causal=False)


@pytest.mark.parametrize("N", [1, 33, 65, 127])
def test_non_aligned_seqlen_causal(N):
    check_fwd(B=1, H=2, N=N, D=64, causal=True)


# ── Batch = 1, head = 1 ──────────────────────────────────────────────

def test_single_batch_single_head():
    check_fwd(B=1, H=1, N=128, D=64, causal=False)
    check_fwd(B=1, H=1, N=128, D=64, causal=True)


# ── Varied head_dim ──────────────────────────────────────────────────

@pytest.mark.parametrize("D", [16, 32, 64, 128])
def test_varied_head_dim(D):
    check_fwd(B=2, H=4, N=128, D=D, causal=False)


# ── Very short sequences ────────────────────────────────────────────

@pytest.mark.parametrize("N", [1, 2, 4, 8, 16])
def test_very_short_seq(N):
    check_fwd(B=2, H=2, N=N, D=64, causal=False)


# ── Large batch / many heads ────────────────────────────────────────

def test_large_batch():
    check_fwd(B=16, H=1, N=64, D=64, causal=False)


def test_many_heads():
    check_fwd(B=1, H=32, N=64, D=64, causal=False)


# ── Naive also works on edge cases ──────────────────────────────────

@pytest.mark.parametrize("N", [1, 33, 65])
def test_naive_edge(N):
    torch.manual_seed(42)
    B, H, D = 1, 2, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    O_ref = ref_attention(Q, K, V).half()
    O_naive = flash_attn_cuda.naive_fwd(Q, K, V, causal=False)

    rel_err = ((O_naive.float() - O_ref.float()).abs().max().item()
               / (O_ref.float().abs().max().item() + 1e-5))
    assert rel_err < 2e-2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
