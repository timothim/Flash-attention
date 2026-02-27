"""
Tests for the Triton Flash Attention implementation.
Compares forward and backward against PyTorch SDPA in float32 as reference.
"""

import pytest
import torch
import math

from flash_attn_triton import flash_attn_triton_fwd, flash_attn_triton_bwd


def ref_attention(Q, K, V, causal=False):
    """Standard attention in float32 for reference."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V.float())
    return O


# ── Forward tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (2, 8, 256, 64),
    (1, 4, 512, 128),
    (2, 4, 1024, 64),
])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_fwd(B, H, N, D, causal):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    O_ref = ref_attention(Q, K, V, causal=causal).half()
    O_tri, _ = flash_attn_triton_fwd(Q, K, V, causal=causal)

    max_err = (O_tri.float() - O_ref.float()).abs().max().item()
    rel_err = max_err / (O_ref.float().abs().max().item() + 1e-5)
    print(f"Triton fwd  B={B} H={H} N={N} D={D} causal={causal}  "
          f"max_err={max_err:.4e}  rel_err={rel_err:.4e}")
    assert rel_err < 2e-2, f"Relative error too high: {rel_err}"


# ── Backward tests ────────────────────────────────────────────────────

@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 64, 64),
    (2, 4, 128, 64),
    (1, 4, 256, 128),
])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_bwd(B, H, N, D, causal):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    # Reference backward via autograd.
    Q_ref = Q.detach().float().requires_grad_(True)
    K_ref = K.detach().float().requires_grad_(True)
    V_ref = V.detach().float().requires_grad_(True)
    O_ref = ref_attention(Q_ref, K_ref, V_ref, causal=causal)
    O_ref.backward(dO.float())

    # Our Triton backward.
    O_tri, L_tri = flash_attn_triton_fwd(Q, K, V, causal=causal)
    dQ_tri, dK_tri, dV_tri = flash_attn_triton_bwd(
        Q, K, V, O_tri, dO, L_tri, causal=causal)

    for name, ours, ref in [("dQ", dQ_tri, Q_ref.grad),
                             ("dK", dK_tri, K_ref.grad),
                             ("dV", dV_tri, V_ref.grad)]:
        max_err = (ours.float() - ref.float()).abs().max().item()
        ref_scale = ref.float().abs().max().item() + 1e-5
        rel_err = max_err / ref_scale
        print(f"  {name}:  max_err={max_err:.4e}  rel_err={rel_err:.4e}")
        assert rel_err < 0.1, f"{name} relative error too high: {rel_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
