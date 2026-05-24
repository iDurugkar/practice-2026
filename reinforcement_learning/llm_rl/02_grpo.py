"""
Assignment 2: GRPO — Group Relative Policy Optimization
========================================================
GRPO (DeepSeek-R1, Shao et al. 2024) replaces the value network in PPO with
group-relative advantage normalization, making it critic-free and well-suited
to LLM training. It is the algorithm behind DeepSeek-R1's reasoning improvements.

Key insight: for a given prompt, sample G outputs, score all with a reward model,
then normalize rewards within the group to estimate advantages — no value network needed.

Topics:
  - GRPO objective and its derivation from PPO
  - Group advantage estimation: Â_i = (r_i - mean(r)) / std(r)
  - Why critic-free matters for LLMs (memory, compute, stability)
  - Token-level loss masking (only generated tokens, not prompt)
  - Clip ratio and KL penalty (both can be used)
  - Comparison: GRPO vs. PPO vs. REINFORCE with baseline

Paper refs:
  - DeepSeekMath / GRPO (Shao et al. 2024)  https://arxiv.org/abs/2402.03300
  - DeepSeek-R1 (DeepSeek-AI 2025)  https://arxiv.org/abs/2501.12948
  - REINFORCE with baseline (Williams 1992)

Setup: pip install torch numpy matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Re-use the toy arithmetic environment from assignment 01
# (copy the relevant pieces or import from there)
import random

VOCAB = list("0123456789+-* =<>")
TOK2ID = {c: i for i, c in enumerate(VOCAB)}
PAD_ID = len(VOCAB)
EOS_ID = TOK2ID[">"]
BOS_ID = TOK2ID["<"]
VOCAB_SIZE = len(VOCAB) + 1
MAX_LEN = 12


def expression_reward(tokens: list[int]) -> float:
    text = "".join(VOCAB[t] if t < len(VOCAB) else "" for t in tokens
                   if t not in (BOS_ID, EOS_ID, PAD_ID))
    text = text.strip()
    try:
        lhs, rhs = text.split("=")
        return float(abs(eval(lhs.strip()) - int(rhs.strip())) < 0.5)  # noqa: S307
    except Exception:
        return 0.0


def format_reward(tokens: list[int]) -> float:
    """Bonus reward for correct format: 'A op B = C' regardless of correctness."""
    text = "".join(VOCAB[t] if t < len(VOCAB) else "" for t in tokens
                   if t not in (BOS_ID, EOS_ID, PAD_ID))
    return float("=" in text and any(op in text for op in ["+", "-", "*"]))


# ---------------------------------------------------------------------------
# Part 1: GRPO advantage estimation
# ---------------------------------------------------------------------------

def group_relative_advantage(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Given rewards for G outputs from the same prompt:
      Â_i = (r_i - mean(r)) / (std(r) + ε)

    Args:
      rewards: shape (G,) — one scalar reward per output in the group.
    Returns:
      advantages: shape (G,) — normalized, zero-mean within the group.

    TODO: Implement. Note: if all rewards are equal (std ≈ 0), advantages → 0.
    This is the key design choice: advantages are relative, not absolute.
    """
    raise NotImplementedError


def group_relative_advantage_batched(rewards: Tensor) -> Tensor:
    """
    Batched version: rewards shape (B, G) where B=num_prompts, G=group_size.
    Normalize within each group independently.
    Returns advantages of shape (B, G).

    TODO: Implement. This is what you'd use in a real training loop.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    log_probs: Tensor,
    log_probs_ref: Tensor,
    advantages: Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[Tensor, dict]:
    """
    GRPO objective (two variants — implement both):

    Variant A (clip-based, like PPO):
      r_t = exp(log_π_θ - log_π_ref)  [importance ratio against reference]
      L = -E[min(r_t · Â, clip(r_t, 1-ε, 1+ε) · Â)]

    Variant B (KL-penalty, simpler):
      L = -E[log_π_θ · Â] + β · KL(π_θ || π_ref)
      KL ≈ mean(log_π_θ - log_π_ref)

    Args:
      log_probs: (B*G, T) — token log-probs from current policy (masked to output only)
      log_probs_ref: (B*G, T) — token log-probs from reference policy
      advantages: (B*G,) — group-relative advantages, one per sequence

    TODO: Implement both variants. Return (loss, metrics_dict).
    metrics_dict should include: clip_fraction, mean_kl, mean_advantage.

    Important: average log_probs over the *output* token dimension only
    (mask out prompt tokens — they don't affect the policy gradient).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: Group sampling rollout
# ---------------------------------------------------------------------------

class ToyLM(nn.Module):
    """GPT-like LM (same architecture as in assignment 01, minimal version)."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(MAX_LEN + 4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=128, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(pos)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)

    @torch.no_grad()
    def generate_group(self, prompt: Tensor, G: int, max_new: int = MAX_LEN,
                       temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Sample G completions for a single prompt (expand to batch of G).
        Return (token_ids, log_probs) both of shape (G, max_new).

        TODO: Implement. Store per-token log-probs for the generated tokens only.
        These are needed for the GRPO loss.
        """
        raise NotImplementedError

    def log_probs_for_tokens(self, tokens: Tensor, generated_start: int) -> Tensor:
        """
        Forward pass on `tokens` (prompt + generated), return log-probs of the
        generated portion only. Shape: (B, T_generated).

        TODO: Implement. Used to recompute log_probs for the current θ during update.
        """
        raise NotImplementedError


def collect_group_rollouts(
    policy: ToyLM, ref_policy: ToyLM,
    prompts: list[Tensor], G: int,
    reward_fns: list,
) -> dict:
    """
    For each prompt, generate G completions and score with all reward_fns.
    Compute group-relative advantages per reward function, then combine.

    Args:
      prompts: list of prompt tensors (each shape (1, T_prompt))
      G: group size (typically 4–16 in GRPO)
      reward_fns: list of callable(tokens) → float  [can combine multiple rewards]

    Return dict with:
      all_tokens: (B*G, T)
      prompt_len: int
      log_probs_old: (B*G, T_generated)
      log_probs_ref: (B*G, T_generated)
      advantages: (B*G,)
      rewards: (B*G,)

    TODO: Implement. The batching here is the engineering challenge of GRPO at scale.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Full GRPO training loop
# ---------------------------------------------------------------------------

def grpo_train(
    n_steps: int = 1000,
    G: int = 8,
    n_prompts_per_step: int = 8,
    kl_beta: float = 0.04,
    clip_eps: float = 0.2,
    lr: float = 5e-5,
) -> list[dict]:
    """
    Full GRPO training. Track: mean_reward, mean_kl, clip_fraction, accuracy.

    Prompts: random BOS tokens (the model learns to generate valid equations from scratch).

    TODO:
      1. Initialize policy and reference policy (same weights).
      2. Each step: collect_group_rollouts → compute grpo_loss → update policy.
      3. Reference policy stays frozen.
      4. Log metrics every 50 steps.
      5. Return list of log dicts.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5: Comparison — GRPO vs REINFORCE-with-baseline vs PPO
# ---------------------------------------------------------------------------

def reinforce_with_baseline(
    policy: ToyLM, optimizer: optim.Optimizer,
    n_rollouts: int = 64,
) -> dict:
    """
    REINFORCE with a running mean baseline:
      L = -E[(r - b) · log π(a|s)]
      b ← 0.99 b + 0.01 mean(r)   [exponential moving average]

    TODO: Implement. This is the simplest policy gradient method —
    compare its variance to GRPO's group-relative normalization.
    """
    raise NotImplementedError


def compare_algorithms():
    """
    Compare on toy arithmetic task for 500 training steps:
      - GRPO (G=8)
      - GRPO (G=2) — small group, high variance
      - REINFORCE with baseline
      - REINFORCE without baseline (pure Monte Carlo)

    Plot:
      (a) Mean accuracy vs. steps
      (b) Gradient variance (estimated) vs. steps — key motivation for GRPO
      (c) Mean advantage magnitude vs. steps

    The point: GRPO's group normalization dramatically reduces variance
    compared to REINFORCE, without needing a separate value network.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 6: Ablation — group size G
# ---------------------------------------------------------------------------

def group_size_ablation(G_values: list[int] = [1, 2, 4, 8, 16]):
    """
    Train GRPO with different group sizes G.
    G=1: equivalent to REINFORCE (no group normalization possible, std=0).
    G=∞: oracle (estimates true advantage via large sample).

    TODO:
      1. Train for 300 steps with each G.
      2. Plot: accuracy and gradient variance vs. G (at step 300).
      3. What's the practical sweet spot? (Empirically usually G=4-8.)
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== GRPO Training ===")
    logs = grpo_train(n_steps=500, G=8)

    print("\n=== Algorithm Comparison ===")
    compare_algorithms()

    print("\n=== Group Size Ablation ===")
    group_size_ablation()
