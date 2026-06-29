"""
Assignment 3: Attention From Scratch & the Online-Softmax Trick
===============================================================
Attention is just a numerically careful softmax over scores, plus some
reshaping for multiple heads. Implement it from the ground up, then implement
the "online softmax" that produces the same result in a single streaming pass
over key/value blocks — the core idea that makes FlashAttention memory-efficient.

Topics:
  - Numerically stable softmax (subtract the row max)
  - Scaled dot-product attention with an optional causal mask
  - Multi-head attention via reshape + matmul
  - Online (streaming) softmax: running max and denominator over blocks

Refs:
  - Attention is all you need (Vaswani et al. 2017)  https://arxiv.org/abs/1706.03762
  - FlashAttention / online softmax (Dao et al. 2022)  https://arxiv.org/abs/2205.14135

Setup: pip install torch   (in the lean env)
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Part 1: Numerically stable softmax
# ---------------------------------------------------------------------------


def stable_softmax(scores: Tensor, dim: int = -1) -> Tensor:
    """
    Softmax along `dim` that does not overflow for large scores.

    TODO: subtract scores.max(dim, keepdim=True).values before exp, then
    divide by the sum along `dim`. (Subtracting a constant per row leaves the
    softmax unchanged but keeps exp() in range.)
    """
    shifted_scores = scores - scores.max(dim, keepdim=True).values
    exps = shifted_scores.exp()
    return exps / exps.sum(dim, keepdim=True)


# ---------------------------------------------------------------------------
# Part 2: Scaled dot-product attention
# ---------------------------------------------------------------------------


def sdpa(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
    """
    Scaled dot-product attention. q, k, v: (..., L, d). Returns (..., L, d).
        scores = q @ k.transpose(-2, -1) / sqrt(d)
        if causal: set scores[..., i, j] = -inf for j > i
        out = stable_softmax(scores) @ v

    TODO: implement using stable_softmax. For the causal mask build a boolean
    (L, L) upper-triangular mask via torch.triu(torch.ones(L, L, bool), diagonal=1)
    and scores.masked_fill(mask, float('-inf')).
    """
    d = k.shape[-1]
    L = k.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        mask = torch.triu(
            torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    out = stable_softmax(scores) @ v
    return out


# ---------------------------------------------------------------------------
# Part 3: Multi-head attention
# ---------------------------------------------------------------------------


def multihead_attention(
    x: Tensor, Wqkv: Tensor, Wo: Tensor, n_heads: int, causal: bool = False
) -> Tensor:
    """
    x: (B, L, D). Wqkv: (D, 3D) projects to concatenated q, k, v. Wo: (D, D).
    Split into n_heads of size d = D // n_heads, run sdpa per head, concat, project.

    TODO:
      - qkv = x @ Wqkv ; split last dim into q, k, v each (B, L, D)
      - reshape each to (B, n_heads, L, d): view to (B, L, n_heads, d) then
        transpose dims 1 and 2
      - out = sdpa(q, k, v, causal)              # (B, n_heads, L, d)
      - merge heads back to (B, L, D) (transpose back, reshape), then @ Wo
    """
    assert len(x.shape) == 3
    B, L, D = x.shape
    qkv = x @ Wqkv
    q, k, v = qkv[..., :D], qkv[..., D : 2 * D], qkv[..., -D:]
    q = q.reshape([*q.shape[:-1], n_heads, D // n_heads]).transpose(1, 2)
    k = k.reshape([*k.shape[:-1], n_heads, D // n_heads]).transpose(1, 2)
    v = v.reshape([*v.shape[:-1], n_heads, D // n_heads]).transpose(1, 2)

    out = sdpa(q, k, v, causal=causal)
    out = out.transpose(1, 2)
    out = out.reshape([*out.shape[:-2], D])
    return out @ Wo


# ---------------------------------------------------------------------------
# Part 4 (advanced): Online softmax attention in a single streaming pass
# ---------------------------------------------------------------------------


def online_softmax_attention(
    q: Tensor, k: Tensor, v: Tensor, block: int = 16
) -> Tensor:
    """
    Compute (non-causal) attention output WITHOUT ever materializing the full
    (L, L) score matrix, by streaming over key/value blocks while keeping a
    running max `m`, running denominator `l`, and running weighted output `acc`.
    q, k, v: (L, d). Return (L, d). This is the FlashAttention recurrence.

    Initialize m = -inf (L,), l = 0 (L,), acc = 0 (L, d). For each block
    (Kb, Vb) of keys/values:
        s     = q @ Kb.transpose(-2, -1) / sqrt(d)     # (L, block)
        m_new = torch.maximum(m, s.max(dim=-1).values) # (L,)
        corr  = (m - m_new).exp()                      # rescale prior state
        p     = (s - m_new[:, None]).exp()             # (L, block)
        l     = l * corr + p.sum(dim=-1)
        acc   = acc * corr[:, None] + p @ Vb
        m     = m_new
    return acc / l[:, None]

    TODO: implement the recurrence above (iterate with k.split(block) and
    v.split(block)).
    """
    L, d = q.shape
    m = torch.full((L,), float("-inf"))
    l = torch.zeros_like(m)
    acc = torch.zeros((L, d))
    for Kb, Vb in zip(k.split(block), v.split(block)):
        s = q @ Kb.transpose(-2, -1) / math.sqrt(d)
        m_new = torch.maximum(m, s.max(dim=-1).values)
        corr = (m - m_new).exp()
        p = (s - m_new[:, None]).exp()
        l = l * corr + p.sum(dim=-1)
        acc = acc * corr[:, None] + p @ Vb
        m = m_new
    return acc / l[:, None]


if __name__ == "__main__":
    torch.manual_seed(0)

    # Part 1: stable even with huge scores.
    s = torch.randn(3, 5) * 50
    print(
        "softmax max err vs torch:",
        (stable_softmax(s) - torch.softmax(s, -1)).abs().max().item(),
    )

    # Part 2: check against torch's fused SDPA (its default scale is 1/sqrt(d)).
    q, k, v = (torch.randn(2, 7, 8) for _ in range(3))
    print(
        "sdpa max err:",
        (sdpa(q, k, v) - F.scaled_dot_product_attention(q, k, v)).abs().max().item(),
    )
    ref_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(
        "causal sdpa max err:", (sdpa(q, k, v, causal=True) - ref_c).abs().max().item()
    )

    # Part 3: multi-head — check output shape.
    B, L, D, H = 2, 6, 16, 4
    x = torch.randn(B, L, D)
    Wqkv, Wo = torch.randn(D, 3 * D) * 0.1, torch.randn(D, D) * 0.1
    out = multihead_attention(x, Wqkv, Wo, n_heads=H)
    print("multihead output shape:", tuple(out.shape), "(expect (2, 6, 16))")

    # Part 4: online softmax must match the dense result.
    q1, k1, v1 = (torch.randn(20, 8) for _ in range(3))
    dense = sdpa(q1, k1, v1)
    print(
        "online vs dense max err:",
        (online_softmax_attention(q1, k1, v1, block=6) - dense).abs().max().item(),
    )
