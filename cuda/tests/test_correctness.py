"""
Correctness tests for the CUDA Flash Attention kernels.

Compares forward and backward outputs against PyTorch's scaled_dot_product_attention
(computed in float32 as the reference).
"""

import pytest
import torch
import math

import flash_attn_cuda


def ref_attention(Q, K, V, causal=False):
    """Standard attention in float32 — gold reference."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V.float())
    return O


# ═══════════════════════════════════════════════════════════════════════
# Forward — Naive baseline
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (1, 2, 256, 128),
])
@pytest.mark.parametrize("causal", [False, True])
def test_naive_fwd(B, H, N, D, causal):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    O_ref = ref_attention(Q, K, V, causal=causal).half()
    O_naive = flash_attn_cuda.naive_fwd(Q, K, V, causal=causal)

    max_err = (O_naive.float() - O_ref.float()).abs().max().item()
    ref_scale = O_ref.float().abs().max().item() + 1e-5
    rel_err = max_err / ref_scale
    print(f"naive_fwd  B={B} H={H} N={N} D={D} causal={causal}  "
          f"max_err={max_err:.4e}  rel_err={rel_err:.4e}")
    assert rel_err < 2e-2, f"Relative error too high: {rel_err}"


# ═══════════════════════════════════════════════════════════════════════
# Forward — Flash Attention
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (2, 8, 256, 64),
    (1, 4, 512, 128),
    (2, 4, 1024, 64),
])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_fwd(B, H, N, D, causal):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    O_ref = ref_attention(Q, K, V, causal=causal).half()
    O_flash, L = flash_attn_cuda.flash_fwd(Q, K, V, causal=causal)

    max_err = (O_flash.float() - O_ref.float()).abs().max().item()
    ref_scale = O_ref.float().abs().max().item() + 1e-5
    rel_err = max_err / ref_scale
    print(f"flash_fwd  B={B} H={H} N={N} D={D} causal={causal}  "
          f"max_err={max_err:.4e}  rel_err={rel_err:.4e}")
    assert rel_err < 2e-2, f"Relative error too high: {rel_err}"


# ═══════════════════════════════════════════════════════════════════════
# Backward — Flash Attention
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (1, 2, 256, 128),
])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_bwd(B, H, N, D, causal):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    # Reference backward via autograd in float32.
    Q_ref = Q.detach().float().requires_grad_(True)
    K_ref = K.detach().float().requires_grad_(True)
    V_ref = V.detach().float().requires_grad_(True)
    O_ref = ref_attention(Q_ref, K_ref, V_ref, causal=causal)
    O_ref.backward(dO.float())

    # Our CUDA backward.
    O_flash, L = flash_attn_cuda.flash_fwd(Q, K, V, causal=causal)
    dQ, dK, dV = flash_attn_cuda.flash_bwd(Q, K, V, O_flash, dO, L, causal=causal)

    for name, ours, ref in [("dQ", dQ, Q_ref.grad),
                             ("dK", dK, K_ref.grad),
                             ("dV", dV, V_ref.grad)]:
        max_err = (ours.float() - ref.float()).abs().max().item()
        ref_scale = ref.float().abs().max().item() + 1e-5
        rel_err = max_err / ref_scale
        print(f"  {name}:  max_err={max_err:.4e}  rel_err={rel_err:.4e}")
        assert rel_err < 0.1, f"{name} relative error too high: {rel_err}"


# ═══════════════════════════════════════════════════════════════════════
# Flash fwd matches Naive fwd
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B,H,N,D", [
    (2, 4, 128, 64),
    (1, 2, 256, 128),
])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_vs_naive(B, H, N, D, causal):
    torch.manual_seed(7)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    O_naive = flash_attn_cuda.naive_fwd(Q, K, V, causal=causal)
    O_flash, _ = flash_attn_cuda.flash_fwd(Q, K, V, causal=causal)

    max_err = (O_flash.float() - O_naive.float()).abs().max().item()
    print(f"flash_vs_naive  N={N} D={D} causal={causal}  max_err={max_err:.4e}")
    assert max_err < 0.05, f"Max error too high: {max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
