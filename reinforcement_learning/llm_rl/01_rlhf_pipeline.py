"""
Assignment 1: RLHF Pipeline — Reward Model + PPO Fine-Tuning
=============================================================
Implement the three-stage RLHF pipeline from scratch on a toy problem:
  Stage 1: SFT (supervised fine-tuning) — already done, we start from an SFT model
  Stage 2: Reward model training from human preferences
  Stage 3: PPO fine-tuning with KL penalty toward reference policy

The toy task: a language model that generates arithmetic expressions.
The "reward" is mathematical correctness — a clean stand-in for human preference
that lets you verify the pipeline works without LLM API calls.

Topics:
  - Bradley-Terry reward model and listwise extensions
  - KL-regularized PPO: r_total = r_rm - β KL(π_θ || π_ref)
  - Token-level KL vs. sequence-level KL
  - Reward hacking: what happens as β → 0?
  - Reference policy freezing and why it's critical

Paper refs:
  - InstructGPT (Ouyang et al. 2022)  https://arxiv.org/abs/2203.02155
  - RLHF (Ziegler et al. 2019)  https://arxiv.org/abs/1909.08593
  - Secrets of RLHF (Zheng et al. 2023)  https://arxiv.org/abs/2307.04964

Setup: pip install torch numpy matplotlib transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch import Tensor
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Toy LM: generates arithmetic expressions like "3 + 4 = 7"
# The vocabulary is digits 0-9, operators +/-/*, equals, space, <eos>.
# "Reward" = 1 if the equation is mathematically correct, else 0.
# This gives us a ground-truth reward signal for verification.
# ---------------------------------------------------------------------------

VOCAB = list("0123456789+-* =<>")
TOK2ID = {c: i for i, c in enumerate(VOCAB)}
PAD_ID = len(VOCAB)
EOS_ID = TOK2ID[">"]
BOS_ID = TOK2ID["<"]
VOCAB_SIZE = len(VOCAB) + 1  # +1 for PAD

MAX_LEN = 12  # "3 + 4 = 7" → 9 tokens


def expression_reward(tokens: list[int]) -> float:
    """
    Parse a generated sequence and return 1.0 if it's a correct equation, 0.0 otherwise.
    Expected format: "A op B = C" where op ∈ {+, -, *}.
    """
    text = "".join(VOCAB[t] if t < len(VOCAB) else "" for t in tokens
                   if t not in (BOS_ID, EOS_ID, PAD_ID))
    text = text.strip()
    try:
        lhs, rhs = text.split("=")
        return float(abs(eval(lhs.strip()) - int(rhs.strip())) < 0.5)  # noqa: S307
    except Exception:
        return 0.0


def sample_preference_pair() -> tuple[list[int], list[int], int]:
    """
    Generate a (chosen, rejected) preference pair with known ground-truth winner.
    chosen: a correct equation (reward=1)
    rejected: an incorrect equation (reward=0)
    Returns (chosen_tokens, rejected_tokens, winner=0 meaning chosen wins)
    """
    a, b = random.randint(1, 9), random.randint(1, 9)
    op = random.choice(["+", "-", "*"])
    correct_c = eval(f"{a}{op}{b}")  # noqa: S307
    if not (0 <= correct_c <= 9):
        return sample_preference_pair()
    wrong_c = (correct_c + random.randint(1, 5)) % 10
    chosen = f"{a} {op} {b} = {correct_c}"
    rejected = f"{a} {op} {b} = {wrong_c}"
    enc = lambda s: [BOS_ID] + [TOK2ID[c] for c in s if c in TOK2ID] + [EOS_ID]
    return enc(chosen), enc(rejected), 0


# ---------------------------------------------------------------------------
# Part 1: Reward Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Transformer encoder → scalar reward head.
    Takes token sequence, outputs a single scalar R(sequence).
    Trained to satisfy: R(chosen) > R(rejected) via Bradley-Terry loss.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2, max_len: int = MAX_LEN + 2):
        super().__init__()
        # TODO: Token embedding + positional embedding + TransformerEncoder
        # Final: mean-pool over sequence dim → Linear(d_model, 1)
        raise NotImplementedError

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return scalar reward of shape (B,)."""
        raise NotImplementedError


def bradley_terry_loss(reward_chosen: Tensor, reward_rejected: Tensor) -> Tensor:
    """
    BT loss: L = -E[log σ(R(chosen) - R(rejected))]
    Also compute and return accuracy = mean(R(chosen) > R(rejected)).

    TODO: Implement.
    """
    raise NotImplementedError


def train_reward_model(n_pairs: int = 5000, n_epochs: int = 5) -> RewardModel:
    """
    TODO:
      1. Sample n_pairs preference pairs.
      2. Train RewardModel to minimize BT loss.
      3. Log: loss and accuracy per epoch.
      4. Verify: reward of correct equations > incorrect equations on held-out set.
      5. Return trained model.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Reference Policy (SFT model)
# ---------------------------------------------------------------------------

class LanguageModel(nn.Module):
    """Simple GPT-like decoder-only Transformer for toy arithmetic."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2, max_len: int = MAX_LEN + 2):
        super().__init__()
        # TODO: Causal Transformer decoder
        # Embedding → TransformerDecoder (causal mask) → Linear(d_model, vocab_size)
        raise NotImplementedError

    def forward(self, tokens: Tensor) -> Tensor:
        """Return logits of shape (B, T, V)."""
        raise NotImplementedError

    def generate(self, prompt: Tensor, max_new: int = MAX_LEN,
                 temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Autoregressive generation. Return (token_ids, log_probs) where
        log_probs[t] = log π(a_t | a_{<t}, prompt).
        """
        raise NotImplementedError


def pretrain_sft(n_steps: int = 10_000) -> LanguageModel:
    """
    Pretrain LM on correct equations (SFT stage).
    After training, it should generate correct equations ~80% of the time.
    TODO: Implement next-token prediction loss. Return trained model.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: PPO with KL penalty (RLHF stage)
# ---------------------------------------------------------------------------

def compute_kl_penalty(policy: LanguageModel, ref_policy: LanguageModel,
                       tokens: Tensor) -> Tensor:
    """
    Token-level KL: KL(π_θ || π_ref) = Σ_t π_θ(a_t|·) log [π_θ(a_t|·) / π_ref(a_t|·)]
    Approximated as: Σ_t [log π_θ(a_t) - log π_ref(a_t)]

    TODO: Compute forward KL using log-probs from both models on the same tokens.
    Return per-sequence KL of shape (B,).
    """
    raise NotImplementedError


def rlhf_ppo_step(
    policy: LanguageModel,
    ref_policy: LanguageModel,
    reward_model: RewardModel,
    optimizer: optim.Optimizer,
    n_rollouts: int = 64,
    kl_beta: float = 0.1,
    clip_eps: float = 0.2,
    gamma: float = 1.0,
) -> dict:
    """
    One RLHF-PPO step:
      1. Sample n_rollouts sequences from current policy (starting from BOS).
      2. Score each with reward_model (sequence-level reward at EOS).
      3. Compute KL penalty per token, subtract: r_total = r_rm - β * KL
      4. Compute advantages (for sequence-level reward: just the total).
      5. PPO-clip update on log_probs.

    Return dict: {mean_reward, mean_kl, policy_loss, mean_correct}.
    mean_correct tracks ground-truth correctness (not RM score).

    TODO: Implement.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Reward hacking experiment
# ---------------------------------------------------------------------------

def reward_hacking_experiment(kl_betas: list[float] = [0.0, 0.01, 0.1, 0.5]):
    """
    Train RLHF-PPO with different KL penalties β.
    Track both RM score and ground-truth accuracy over training.

    Expected observation:
      - β=0.0: RM score rises fast, but true accuracy eventually drops (reward hacking)
      - β=0.5: stable improvement, RM score and accuracy track together
      - Optimal β somewhere in between

    TODO:
      1. For each β, train for 1000 steps.
      2. Plot RM score and ground-truth accuracy on same axes (different lines).
      3. Identify the β where hacking is most visible.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("Stage 1: SFT pretraining...")
    ref_policy = pretrain_sft(n_steps=5000)

    print("\nStage 2: Reward model training...")
    reward_model = train_reward_model(n_pairs=3000)

    print("\nStage 3: RLHF fine-tuning...")
    policy = type(ref_policy)()  # fresh policy same architecture
    policy.load_state_dict(ref_policy.state_dict())  # init from SFT
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    logs = []
    for step in range(500):
        log = rlhf_ppo_step(policy, ref_policy, reward_model, optimizer,
                            kl_beta=0.1, n_rollouts=64)
        logs.append(log)
        if step % 50 == 0:
            print(f"Step {step}: reward={log['mean_reward']:.3f} "
                  f"kl={log['mean_kl']:.3f} acc={log['mean_correct']:.3f}")

    print("\nReward Hacking Experiment...")
    reward_hacking_experiment()
