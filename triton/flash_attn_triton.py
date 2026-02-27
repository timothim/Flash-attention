"""
Flash Attention 2 — Triton implementation (forward + backward).

Mirrors the CUDA kernel logic but leverages Triton's automatic shared-memory
management, register allocation, and autotuning.
"""

import math
import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel
# ═══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 128}, num_warps=8, num_stages=3),
    ],
    key=["N", "D"],
)
@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb,
    N: tl.constexpr, D: tl.constexpr,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    block_q_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    q_offsets = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, D)

    # Load Q block into SRAM (stays for entire KV loop).
    q_ptrs = (Q_ptr + bh_idx * stride_qb
              + q_offsets[:, None] * stride_qn
              + d_offsets[None, :] * stride_qd)
    q_mask = q_offsets[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Accumulators.
    o_acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_Q], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Upper bound on KV blocks to iterate.
    kv_blocks = tl.cdiv(N, BLOCK_KV)
    if CAUSAL:
        kv_blocks = tl.minimum(kv_blocks,
                               tl.cdiv((block_q_idx + 1) * BLOCK_Q, BLOCK_KV))

    for kv_idx in range(0, kv_blocks):
        kv_offsets = kv_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets[:, None] < N

        # Load K, V.
        k_ptrs = (K_ptr + bh_idx * stride_kb
                  + kv_offsets[:, None] * stride_kn
                  + d_offsets[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        v_ptrs = (V_ptr + bh_idx * stride_vb
                  + kv_offsets[:, None] * stride_vn
                  + d_offsets[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # GEMM-I:  S = Q @ K^T * scale.
        s = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_Q, BLOCK_KV]

        # Boundary mask.
        s = tl.where(kv_offsets[None, :] < N, s, float("-inf"))

        # Causal mask.
        if CAUSAL:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online softmax.
        m_block = tl.max(s, axis=1)  # [BLOCK_Q]
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)

        # GEMM-II:  O += P @ V.
        o_acc = alpha[:, None] * o_acc + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # Normalise.
    o = o_acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    # Store O.
    o_ptrs = (O_ptr + bh_idx * stride_ob
              + q_offsets[:, None] * stride_on
              + d_offsets[None, :] * stride_od)
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < N)

    # Store L (logsumexp).
    l_ptrs = L_ptr + bh_idx * stride_lb + q_offsets
    tl.store(l_ptrs, lse, mask=q_offsets < N)


# ═══════════════════════════════════════════════════════════════════════
# Backward kernel
# ═══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=2),
    ],
    key=["N", "D"],
)
@triton.jit
def _flash_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb,
    N: tl.constexpr, D_dim: tl.constexpr,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    """Outer loop on KV blocks, inner loop on Q blocks."""
    kv_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    kv_offsets = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    d_offsets = tl.arange(0, D_dim)
    kv_mask = kv_offsets[:, None] < N

    # Load K, V for this KV block.
    k_ptrs = (K_ptr + bh_idx * stride_kb
              + kv_offsets[:, None] * stride_kn
              + d_offsets[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

    v_ptrs = (V_ptr + bh_idx * stride_vb
              + kv_offsets[:, None] * stride_vn
              + d_offsets[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

    dk_acc = tl.zeros([BLOCK_KV, D_dim], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, D_dim], dtype=tl.float32)

    num_q_blocks = tl.cdiv(N, BLOCK_Q)
    q_start = 0
    if CAUSAL:
        q_start = kv_block_idx * BLOCK_KV // BLOCK_Q

    for q_block_idx in range(q_start, num_q_blocks):
        q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets[:, None] < N

        # Load Q, dO.
        q_ptrs = (Q_ptr + bh_idx * stride_qb
                  + q_offsets[:, None] * stride_qn
                  + d_offsets[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        do_ptrs = (dO_ptr + bh_idx * stride_ob
                   + q_offsets[:, None] * stride_on
                   + d_offsets[None, :] * stride_od)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        # Load L and D_i.
        l_ptrs = L_ptr + bh_idx * stride_lb + q_offsets
        lse = tl.load(l_ptrs, mask=q_offsets < N, other=0.0)
        d_ptrs = D_ptr + bh_idx * stride_lb + q_offsets
        d_i = tl.load(d_ptrs, mask=q_offsets < N, other=0.0)

        # Recompute S = Q @ K^T * scale.
        s = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_Q, BLOCK_KV]

        # Masks.
        s = tl.where(kv_offsets[None, :] < N, s, float("-inf"))
        if CAUSAL:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Recompute P = exp(S - L).
        p = tl.exp(s - lse[:, None])

        # dP = dO @ V^T.
        dp = tl.dot(do, tl.trans(v))  # [BLOCK_Q, BLOCK_KV]

        # dS = P * (dP - D_i).
        ds = p * (dp - d_i[:, None])

        # Accumulate dV += P^T @ dO.
        dv_acc += tl.dot(tl.trans(p), do)

        # Accumulate dK += dS^T @ Q * scale.
        dk_acc += tl.dot(tl.trans(ds), q) * scale

        # dQ += dS @ K * scale — use atomic add.
        dq_contrib = tl.dot(ds, k) * scale  # [BLOCK_Q, D_dim]
        dq_ptrs = (dQ_ptr + bh_idx * stride_qb
                   + q_offsets[:, None] * stride_qn
                   + d_offsets[None, :] * stride_qd)
        tl.atomic_add(dq_ptrs, dq_contrib.to(dQ_ptr.dtype.element_ty),
                       mask=q_mask)

    # Store dK, dV.
    dk_ptrs = (dK_ptr + bh_idx * stride_kb
               + kv_offsets[:, None] * stride_kn
               + d_offsets[None, :] * stride_kd)
    tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=kv_mask)

    dv_ptrs = (dV_ptr + bh_idx * stride_vb
               + kv_offsets[:, None] * stride_vn
               + d_offsets[None, :] * stride_vd)
    tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=kv_mask)


# ═══════════════════════════════════════════════════════════════════════
# Python wrappers
# ═══════════════════════════════════════════════════════════════════════

def flash_attn_triton_fwd(Q, K, V, causal=False):
    """
    Q, K, V : [B, H, N, D]  float16 CUDA tensors.
    Returns  : (O, L)  where O is [B, H, N, D] fp16, L is [B, H, N] fp32.
    """
    B, H, N, D = Q.shape
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    BH = B * H

    # Reshape to [B*H, N, D].
    q = Q.reshape(BH, N, D)
    k = K.reshape(BH, N, D)
    v = V.reshape(BH, N, D)

    o = torch.zeros_like(q)
    lse = torch.empty(BH, N, device=Q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(D)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_Q"]), BH)

    _flash_attn_fwd_kernel[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0),
        N=N, D=D,
        scale=scale,
        CAUSAL=causal,
    )

    return o.reshape(B, H, N, D), lse.reshape(B, H, N)


def flash_attn_triton_bwd(Q, K, V, O, dO, L, causal=False):
    """
    Backward pass.
    Returns (dQ, dK, dV) all [B, H, N, D] fp16.
    """
    B, H, N, D = Q.shape
    BH = B * H
    scale = 1.0 / math.sqrt(D)

    q = Q.reshape(BH, N, D)
    k = K.reshape(BH, N, D)
    v = V.reshape(BH, N, D)
    o = O.reshape(BH, N, D)
    do = dO.reshape(BH, N, D)
    lse = L.reshape(BH, N)

    # Precompute D_i = rowsum(dO * O).
    d_i = (do.float() * o.float()).sum(dim=-1)  # [BH, N]

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_KV"]), BH)

    _flash_attn_bwd_kernel[grid](
        q, k, v, o, do, lse, d_i,
        dq, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0),
        N=N, D_dim=D,
        scale=scale,
        CAUSAL=causal,
    )

    return (dq.reshape(B, H, N, D),
            dk.reshape(B, H, N, D),
            dv.reshape(B, H, N, D))


# ═══════════════════════════════════════════════════════════════════════
# torch.autograd wrapper
# ═══════════════════════════════════════════════════════════════════════

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal):
        O, L = flash_attn_triton_fwd(Q, K, V, causal=causal)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attn_triton_bwd(Q, K, V, O, dO, L,
                                            causal=ctx.causal)
        return dQ, dK, dV, None


def flash_attention_triton(Q, K, V, causal=False):
    """Drop-in replacement for scaled_dot_product_attention (Triton backend)."""
    return FlashAttentionTriton.apply(Q, K, V, causal)
