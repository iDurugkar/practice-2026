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

# Re-use the toy arithmetic environment from assignment 01
# (copy the relevant pieces or import from there)
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rlhf_pipeline import LanguageModel
from torch import Tensor
from torch.distributions import Categorical

from utils import get_device

VOCAB = list("0123456789+-* =<>")
TOK2ID = {c: i for i, c in enumerate(VOCAB)}
PAD_ID = len(VOCAB)
EOS_ID = TOK2ID[">"]
BOS_ID = TOK2ID["<"]
VOCAB_SIZE = len(VOCAB) + 1
MAX_LEN = 12


def expression_reward(tokens: list[int]) -> float:
    text = "".join(
        VOCAB[t] if t < len(VOCAB) else ""
        for t in tokens
        if t not in (BOS_ID, EOS_ID, PAD_ID)
    )
    text = text.strip()
    try:
        lhs, rhs = text.split("=")
        return float(abs(eval(lhs.strip()) - int(rhs.strip())) < 0.5)  # noqa: S307
    except Exception:
        return 0.0


def format_reward(tokens: list[int]) -> float:
    """Bonus reward for correct format: 'A op B = C' regardless of correctness."""
    text = "".join(
        VOCAB[t] if t < len(VOCAB) else ""
        for t in tokens
        if t not in (BOS_ID, EOS_ID, PAD_ID)
    )
    if "=" not in text:
        return 0.0
    else:
        sections = text.split("=")
        if len(sections) > 2:
            return 0.1
        if any(op in sections[1] for op in ["+", "-", "*"]):
            return 0.2
        if not any(op in sections[0] for op in ["+", "-", "*"]):
            return 0.2
        return 1.0


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
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def group_relative_advantage_batched(rewards: Tensor) -> Tensor:
    """
    Batched version: rewards shape (B, G) where B=num_prompts, G=group_size.
    Normalize within each group independently.
    Returns advantages of shape (B, G).

    TODO: Implement. This is what you'd use in a real training loop.
    """
    gra_function = torch.vmap(group_relative_advantage)
    return gra_function(rewards)


# ---------------------------------------------------------------------------
# Part 2: GRPO loss
# ---------------------------------------------------------------------------


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def grpo_loss(
    log_probs: Tensor,
    log_probs_old: Tensor,
    log_probs_ref: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[Tensor, dict]:
    """
    GRPO objective:

    Variant A (clip-based, like PPO):
      r_t = exp(log_π_θ - log_π_old)  [importance ratio against generating]
      KL = exp(logπ_ref − logπ_θ) − (logπ_ref − logπ_θ) − 1
      L = -E[min(r_t · Â, clip(r_t, 1-ε, 1+ε) · Â) -  β · KL(π_θ || π_ref)]

    Args:
      log_probs: (B*G, T) — token log-probs from current policy (masked to output only)
      log_probs_old: (B*G, T) — token log-probs from policy used to generate data
      log_probs_ref: (B*G, T) — token log-probs from reference policy
      advantages: (B*G,) — group-relative advantages, one per sequence
      mask: (B*G, T) -- masking PAD tokens so they aren't used for backprop

    Return (loss, metrics_dict).
    metrics_dict should include: clip_fraction, mean_kl, mean_advantage.

    Note: log_probs will be sent only for outputs
    and PAD_ID tokens will have log_probs set to 0
    """
    log_ratio = log_probs - log_probs_old
    adv = advantages.unsqueeze(-1)

    r_t = torch.exp(log_ratio)
    clip_cond = ((r_t < 1 - clip_eps) | (r_t > 1 + clip_eps)).float()
    clip_frac = masked_mean(clip_cond, mask)

    # http://joschu.net/blog/kl-approx.html
    kl = torch.exp(log_probs_ref - log_probs) - (log_probs_ref - log_probs) - 1
    per_token_loss = (
        torch.minimum(r_t * adv, torch.clip(r_t, 1 - clip_eps, 1 + clip_eps) * adv)
        - kl_beta * kl
    )
    loss = -masked_mean(per_token_loss, mask)

    metrics_dict = {
        "clip_fraction": clip_frac,
        "mean_kl": masked_mean(kl, mask),
        "mean_advantage": advantages.mean(),
        "loss": loss.item(),
    }

    return loss, metrics_dict


# ---------------------------------------------------------------------------
# Part 3: Group sampling rollout
# ---------------------------------------------------------------------------


class ToyLM(nn.Module):
    """GPT-like LM (same architecture as in assignment 01, minimal version)."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(MAX_LEN + 4, d_model)  # Why + 4?
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
    def generate_group(
        self, prompt: Tensor, G: int, max_new: int = MAX_LEN, temperature: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        """
        Sample G completions for a single prompt (expand to batch of G).
        Return (token_ids, log_probs) both of shape (G, max_new).

        TODO: Implement. Store per-token log-probs for the generated tokens only.
        These are needed for the GRPO loss.
        """
        # if we got a single prompt, add a batch dimension
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        # repeat prompt G times
        prompt = prompt.repeat(G, 1)
        # which rollout has finished generation
        finished = torch.zeros((G, 1), device=prompt.device, dtype=torch.bool)
        out_tokens = []
        out_log_probs = []
        for _ in range(max_new):
            # check if output of self.forward is of shape (B*G, T, V)
            logits = self(prompt)[:, -1, :]  # predict just the last one
            log_probs = torch.log_softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(torch.exp(log_probs), num_samples=1)
            # default to PAD_ID if sequence already had a EOS
            tokens = torch.where(
                finished,
                torch.full_like(tokens, PAD_ID),
                tokens,
            )
            lp = torch.gather(log_probs, -1, tokens)

            # if we are using masking do we need this?
            lp = torch.where(
                finished,
                torch.zeros_like(lp),
                lp,
            )

            out_log_probs.append(lp)
            out_tokens.append(tokens)
            finished = finished | (tokens == EOS_ID)
            prompt = torch.cat([prompt, tokens], dim=1)
        return torch.cat(out_tokens, dim=1), torch.cat(out_log_probs, dim=1)

    def log_probs_for_tokens(self, tokens: Tensor, generated_start: int) -> Tensor:
        """
        Forward pass on `tokens` (prompt + generated), return log-probs of the
        generated portion only. Shape: (B, T_generated).

        TODO: Implement. Used to recompute log_probs for the current θ during update.
        """
        logits = self(tokens)[:, :-1, :]  # skip the last one
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob_chosen = torch.gather(
            log_probs, -1, tokens[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return log_prob_chosen[
            :, generated_start - 1 :
        ]  # we already skipped the first token


def collect_group_rollouts(
    policy: ToyLM,
    ref_policy: ToyLM,
    prompts: list[Tensor],
    G: int,
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
      mask: (B*G, T_generated)  mask indicating which tokens are padding

    TODO: Implement. The batching here is the engineering challenge of GRPO at scale.
    """
    prompt_len = prompts[0].shape[-1]
    max_new = MAX_LEN - prompt_len
    assert all(prompt.shape[-1] == prompt_len for prompt in prompts)
    groups = [
        torch.cat(
            [prompt.repeat(G, 1), policy.generate_group(prompt, G, max_new)[0]], dim=1
        )
        for prompt in prompts
    ]
    log_probs_old = [policy.log_probs_for_tokens(group, prompt_len) for group in groups]
    log_probs_ref = [
        ref_policy.log_probs_for_tokens(group, prompt_len) for group in groups
    ]

    # ToDo: try and vectorize this operation
    rewards = torch.tensor(
        [
            [
                sum([reward_fn(seq.tolist()) for reward_fn in reward_fns])
                for seq in group
            ]
            for group in groups
        ],
        device=get_device(),
    )
    advantages = group_relative_advantage_batched(rewards).to(device=get_device())
    all_tokens = torch.cat(groups, dim=0)

    ret_dict = {
        "all_tokens": all_tokens,
        "prompt_len": prompt_len,
        "log_probs_old": torch.cat(log_probs_old, dim=0),
        "log_probs_ref": torch.cat(log_probs_ref, dim=0),
        "advantages": advantages.flatten(),
        "rewards": rewards.flatten(),
        "mask": (all_tokens[:, prompt_len:] != PAD_ID).float(),
    }
    return ret_dict


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

    Prompts: just the BOS token — there is no varying prompt; every group starts
    from `<BOS>` and the model generates the entire equation from scratch.
    Note: with a randomly-initialized policy, the correctness reward is extremely
    sparse (a fresh model almost never emits a valid equation, so all rewards in a
    group are 0 → all advantages are 0 → no gradient). Include `format_reward` in
    `reward_fns` to give a denser shaping signal so learning can get off the ground.

    TODO:
      1. Initialize policy and reference policy (same weights).
      2. Each step: collect_group_rollouts → compute grpo_loss → update policy.
      3. Reference policy stays frozen.
      4. Log metrics every 50 steps.
      5. Return list of log dicts.
    """

    device = get_device()
    policy = ToyLM().to(device=device)
    ref_policy = ToyLM().to(device=device)
    policy.load_state_dict(ref_policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    batch_size = n_prompts_per_step
    group_size = G
    prompts = [torch.tensor([[BOS_ID]], device=device, dtype=torch.long)] * batch_size
    logs = []
    reward_fns = [expression_reward, format_reward]

    for step in range(n_steps):
        optimizer.zero_grad()
        group_dict = collect_group_rollouts(
            policy, ref_policy, prompts, group_size, reward_fns
        )
        loss, metrics = grpo_loss(
            group_dict["log_probs_old"],
            group_dict["log_probs_old"],
            group_dict["log_probs_ref"],
            group_dict["advantages"],
            group_dict["mask"],
            clip_eps,
            kl_beta,
        )
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            text = "".join(
                VOCAB[t] if t < len(VOCAB) else ""
                for t in group_dict["all_tokens"][0].tolist()
                if t not in (BOS_ID, EOS_ID, PAD_ID)
            )
            print(f"{step=}")
            print(f"average_rewards: {torch.mean(group_dict['rewards'])}")
            print(f"mean_KL: {metrics['mean_kl']}")
            print(f"clip_fraction: {metrics['clip_fraction']}")
            print(f"example rollout: {text}")
            print("==============================")
            logs.append(metrics)

    return logs


# ---------------------------------------------------------------------------
# Part 5: Comparison — GRPO vs REINFORCE-with-baseline vs PPO
# ---------------------------------------------------------------------------


def reinforce_loss(
    log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
) -> tuple[Tensor, dict]:
    """

    Args:
      log_probs: (B, T) — token log-probs from current policy (masked to output only)
      advantages: (B,) — reinforce advantages, one per sequence
      mask: (B, T) -- masking PAD tokens so they aren't used for backprop

    Return (loss, metrics_dict).
    metrics_dict should include: clip_fraction, mean_kl, mean_advantage.

    Note: log_probs will be sent only for outputs
    and PAD_ID tokens will have log_probs set to 0
    """

    per_token_loss = advantages.unsqueeze(-1) * log_probs
    loss = -masked_mean(per_token_loss, mask)

    metrics_dict = {
        "mean_advantage": advantages.mean(),
        "loss": loss.item(),
    }

    return loss, metrics_dict


def collect_reinforce_rollouts(
    policy: ToyLM,
    prompts: list[Tensor],
    reward_fns: list,
    baseline: float = 0.0,
    ema_weight: float = 0.01,
) -> dict:
    """
    For each prompt, generate G completions and score with all reward_fns.
    Compute group-relative advantages per reward function, then combine.

    Args:
      prompts: list of prompt tensors (each shape (1, T_prompt))
      reward_fns: list of callable(tokens) → float  [can combine multiple rewards]
      baseline: that should be subtracted from rewards to get advantages
      ema_weight: used to adjust the baseline for the next step

    Return dict with:
      all_tokens: (B, T)
      prompt_len: int
      log_probs: (B, T_generated)
      advantages: (B,)
      rewards: (B,)
      mask: (B, T_generated)  mask indicating which tokens are padding
      baseline: float
    """
    prompt_len = prompts[0].shape[-1]
    max_new = MAX_LEN - prompt_len
    assert all(prompt.shape[-1] == prompt_len for prompt in prompts)
    rollouts = [
        torch.cat([prompt, policy.generate_group(prompt, 1, max_new)[0]], dim=1)
        for prompt in prompts
    ]

    log_probs = [policy.log_probs_for_tokens(ro, prompt_len) for ro in rollouts]

    # ToDo: try and vectorize this operation
    rewards = torch.tensor(
        [
            sum([reward_fn(ro[0].tolist()) for reward_fn in reward_fns])
            for ro in rollouts
        ],
        device=get_device(),
    )
    advantages = rewards - baseline
    all_tokens = torch.cat(rollouts, dim=0)
    baseline = (1 - ema_weight) * baseline + ema_weight * rewards.mean().item()

    ret_dict = {
        "all_tokens": all_tokens,
        "prompt_len": prompt_len,
        "log_probs": torch.cat(log_probs, dim=0),
        "advantages": advantages.flatten(),
        "rewards": rewards.flatten(),
        "mask": (all_tokens[:, prompt_len:] != PAD_ID).float(),
        "baseline": baseline,
    }
    return ret_dict


def reinforce_with_baseline(
    n_steps: int = 1000,
    n_prompts_per_step: int = 8,
    lr: float = 5e-5,
    ema_weight: float = 0.01,
    use_baseline: bool = True,
) -> list[dict]:
    """
    REINFORCE with a running mean baseline:
      L = -E[(r - b) · log π(a|s)]
      b ← 0.99 b + 0.01 mean(r)   [exponential moving average]

    Note: the baseline b is *stateful* — it must persist across calls for the EMA
    to mean anything. A local `b = 0` reset on every step defeats the purpose.
    Thread it through (e.g. a `baseline` arg returned in the dict) or store it as a
    function/module attribute. For the "without baseline" variant in
    compare_algorithms, fix b = 0 (pure Monte-Carlo return).

    TODO: Implement. This is the simplest policy gradient method —
    compare its variance to GRPO's group-relative normalization.
    """
    device = get_device()
    policy = ToyLM().to(device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    batch_size = n_prompts_per_step
    prompts = [torch.tensor([[BOS_ID]], device=device, dtype=torch.long)] * batch_size
    logs = []
    reward_fns = [expression_reward, format_reward]
    baseline = 0.0

    for step in range(n_steps):
        optimizer.zero_grad()
        group_dict = collect_reinforce_rollouts(
            policy, prompts, reward_fns, baseline if use_baseline else 0.0, ema_weight
        )
        baseline = group_dict["baseline"]
        loss, metrics = reinforce_loss(
            group_dict["log_probs"],
            group_dict["advantages"],
            group_dict["mask"],
        )
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            text = "".join(
                VOCAB[t] if t < len(VOCAB) else ""
                for t in group_dict["all_tokens"][0].tolist()
                if t not in (BOS_ID, EOS_ID, PAD_ID)
            )
            print(f"{step=}")
            print(f"average_rewards: {torch.mean(group_dict['rewards'])}")
            print(f"example rollout: {text}")
            print("==============================")
            logs.append(metrics)

    return logs


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
    G=1: degenerate — with a single sample the group mean equals r_i, so every
         advantage is exactly 0 and the policy never updates. This is NOT
         REINFORCE (which uses the raw return r_i); it demonstrates why GRPO needs
         G >= 2 for the group-relative baseline to carry any signal.
    G large: the group estimate of mean/std converges, so the advantage approaches
         the true (oracle) advantage — at the cost of G× more rollouts per step.

    TODO:
      1. Train for 300 steps with each G.
      2. Plot: accuracy and gradient variance vs. G (at step 300).
      3. What's the practical sweet spot? (Empirically usually G=4-8.)
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== GRPO Training ===")
    logs = grpo_train(n_steps=500, G=8)

    print("=== REINFORCE Training ===")
    logs = reinforce_with_baseline(
        n_steps=500,
        n_prompts_per_step=64,
    )

    # print("\n=== Algorithm Comparison ===")
    # compare_algorithms()

    # print("\n=== Group Size Ablation ===")
    # group_size_ablation()
