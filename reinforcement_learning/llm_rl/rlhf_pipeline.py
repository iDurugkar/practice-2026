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

import random
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpmath import eps
from scipy.sparse import data
from torch import Tensor
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from utils import get_device

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

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = MAX_LEN + 2,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )
        self.reward_head = nn.Linear(d_model, 1)
        # Final: mean-pool over sequence dim → Linear(d_model, 1)

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return scalar reward of shape (B,)."""
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device, dtype=torch.long)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.encoer(
            x, src_key_padding_mask=(mask == 0) if mask is not None else None
        )
        x = x.mean(dim=1)
        return self.reward_head(x).squeeze(-1)  # (B,)


def bradley_terry_loss(
    reward_chosen: Tensor, reward_rejected: Tensor
) -> tuple[Tensor, Tensor]:
    """
    BT loss: L = -E[log σ(R(chosen) - R(rejected))]
    Also compute and return accuracy = mean(R(chosen) > R(rejected)).
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return loss, accuracy


def train_reward_model(n_pairs: int = 5000, n_epochs: int = 5) -> RewardModel:
    """
    TODO:
      1. Sample n_pairs preference pairs.
      2. Train RewardModel to minimize BT loss.
      3. Log: loss and accuracy per epoch.
      4. Verify: reward of correct equations > incorrect equations on held-out set.
      5. Return trained model.
    """

    device = get_device()
    reward_model = RewardModel().to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)

    class PrefDataset(Dataset):
        def __init__(self, inp_data: list[tuple[list[int], list[int], int]]) -> None:
            super().__init__()
            self.positives: list[Tensor] = []
            self.negatives: list[Tensor] = []
            self.n_samples: int = len(inp_data)
            for a, b, winner in inp_data:
                a = torch.tensor(a + [PAD_ID] * (MAX_LEN + 2 - len(a)))
                b = torch.tensor(b + [PAD_ID] * (MAX_LEN + 2 - len(b)))
                if winner == 0:
                    self.positives.append(a)
                    self.negatives.append(b)
                else:
                    self.positives.append(b)
                    self.negatives.append(a)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, index) -> tuple[Tensor, Tensor]:
            return self.positives[index], self.negatives[index]

    train_data = [sample_preference_pair() for _ in range(n_pairs)]
    dataset = PrefDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    test_data = [sample_preference_pair() for _ in range(500)]
    test_dataset = PrefDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    for epoch in range(n_epochs):
        for chosen_prompts, rejected_prompts in dataloader:
            optimizer.zero_grad()
            chosen_prompts = chosen_prompts.to(device)
            rejected_prompts = rejected_prompts.to(device)
            loss, _ = bradley_terry_loss(
                reward_model(chosen_prompts), reward_model(rejected_prompts)
            )
            loss.backward()
            optimizer.step()
        accuracy: list[Tensor] = []
        with torch.no_grad():
            for chosen_prompts, rejected_prompts in test_dataloader:
                chosen_prompts = chosen_prompts.to(device)
                rejected_prompts = rejected_prompts.to(device)
                _, acc = bradley_terry_loss(
                    reward_model(chosen_prompts), reward_model(rejected_prompts)
                )
                accuracy.append(acc)
        print(f"Epoch {epoch}, accuracy: {torch.stack(accuracy).mean().item():.3f}")

    return reward_model


# ---------------------------------------------------------------------------
# Part 2: Reference Policy (SFT model)
# ---------------------------------------------------------------------------


def sample_test_example() -> tuple[list[int], list[int]]:
    """
    Generate a (prefix, correct completion) pair used to evaluate the language model
    prefix: the prompt for the equation
    correct completion: the right answer and the EOS token
    """
    a, b = random.randint(1, 9), random.randint(1, 9)
    op = random.choice(["+", "-", "*"])
    correct_c = eval(f"{a}{op}{b}")  # noqa: S307
    if not (0 <= correct_c <= 9):
        return sample_test_example()
    prefix = f"{a} {op} {b} = "
    completion = f"{correct_c}"
    enc_prefix = [BOS_ID] + [TOK2ID[c] for c in prefix if c in TOK2ID]
    enc_completion = [TOK2ID[c] for c in completion if c in TOK2ID] + [EOS_ID]
    return enc_prefix, enc_completion


class LanguageModel(nn.Module):
    """Simple GPT-like decoder-only Transformer for toy arithmetic."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = MAX_LEN + 2,
    ):
        super().__init__()
        # Causal Transformer decoder
        # Embedding → TransformerDecoder (causal mask) → Linear(d_model, vocab_size)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.logits = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return logits of shape (B, T, V)."""
        _, T = tokens.shape
        positions = torch.arange(T, device=tokens.device, dtype=torch.long)
        pos_emb = self.positional_embedding(positions)
        tok_emb = self.token_embedding(tokens)
        x = pos_emb + tok_emb

        mask = torch.triu(
            torch.ones(T, T, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        causal_mask = torch.zeros(T, T, device=tokens.device)
        causal_mask = causal_mask.masked_fill(mask, float("-inf"))
        x = self.decoder(x, mask=causal_mask)
        logits = self.logits(x)
        return logits

    def generate(
        self, prompt: Tensor, max_new: int = MAX_LEN, temperature: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        """
        Autoregressive generation. Return (token_ids, log_probs) where
        log_probs[t] = log π(a_t | a_{<t}, prompt).
        """
        B, _ = prompt.shape
        log_probs_list = []
        output_tokens = []
        tokens = prompt.clone()

        for _ in range(max_new):
            preds = self(tokens)
            logits = preds[:, -1, :] / temperature

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=1)
            log_prob = torch.gather(log_probs, 1, next_tokens)

            tokens = torch.cat([tokens, next_tokens], dim=1)
            log_probs_list.append(log_prob)
            output_tokens.append(next_tokens.clone())
            # If generation is done then break
            if torch.all((next_tokens == PAD_ID) | (next_tokens == EOS_ID)):
                break

        return torch.cat(output_tokens, 1), torch.cat(log_probs_list, 1)


def pretrain_sft(n_steps: int = 10_000) -> LanguageModel:
    """
    Pretrain LM on correct equations (SFT stage).
    After training, it should generate correct equations ~80% of the time.
    """
    device = get_device()
    lm = LanguageModel().to(device=device)
    optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # ignore gradients for PAD

    class PrefDataset(Dataset):
        def __init__(self, inp_data: list[list[int]], test=False) -> None:
            super().__init__()
            self.data: list[Tensor] = []
            self.n_samples: int = len(inp_data)
            for seq in inp_data:
                seq = torch.tensor(seq + [PAD_ID] * (MAX_LEN + 2 - len(seq)))
                self.data.append(seq)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, index) -> Tensor:
            return self.data[index]

    minibatch_size = 50
    train_data = [sample_preference_pair()[0] for _ in range(n_steps * minibatch_size)]
    dataset = PrefDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)
    # test_data = [sample_preference_pair()[0] for _ in range(500)]
    # test_dataset = PrefDataset(test_data)
    # test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    losses = []
    for i, seq in enumerate(dataloader):
        optimizer.zero_grad()
        seq = seq.to(device)
        tr_tokens = seq[:, :-1]
        labels = seq[:, 1:].detach().flatten()
        logits = lm(tr_tokens)
        logits = logits.reshape((-1, logits.shape[-1]))
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Todo evaluate model
        if (i + 1) % 100 == 0:
            lm.eval()
            accuracy: list[float] = []
            # test 100 completions
            with torch.no_grad():
                for _ in range(100):
                    prefix, _ = sample_test_example()
                    gen_completion, _ = lm.generate(
                        torch.tensor([prefix], device=device),
                        max_new=MAX_LEN - len(prefix),
                        temperature=0.5,
                    )
                    full_sequence = prefix + gen_completion[0].cpu().numpy().tolist()
                    accuracy.append(expression_reward(full_sequence))
                print(
                    f"Accuracy at step {i + 1} = {np.mean(accuracy)}, mean loss = {np.mean(losses[-100:])}"
                )
            lm.train()
    return lm


# ---------------------------------------------------------------------------
# Part 3: PPO with KL penalty (RLHF stage)
# ---------------------------------------------------------------------------


def token_logprobs(model: LanguageModel, tokens: Tensor) -> Tensor:
    """log π(a_t | BOS, a_<t) for each position. tokens: (B, T) generated ids (no BOS)."""
    B, T = tokens.shape
    bos = torch.full((B, 1), BOS_ID, device=tokens.device, dtype=tokens.dtype)
    inp = torch.cat([bos, tokens[:, :-1]], dim=1)
    logp = torch.log_softmax(model(inp), dim=-1)  # (B, T, V)
    return torch.gather(logp, -1, tokens.unsqueeze(-1)).squeeze(-1)  # (B, T)


def compute_kl_penalty(
    policy: LanguageModel,
    ref_policy: LanguageModel,
    tokens: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Token-level KL: KL(π_θ || π_ref) = Σ_t π_θ(a_t|·) log [π_θ(a_t|·) / π_ref(a_t|·)]
    Approximated as: Σ_t [log π_θ(a_t) - log π_ref(a_t)]
    Return per-sequence KL of shape (B,).
    """
    B, T = tokens.shape
    log_pi = token_logprobs(policy, tokens)
    with torch.no_grad():
        log_pi_ref = token_logprobs(ref_policy, tokens)
    kl_token = log_pi - log_pi_ref
    if mask is not None:
        kl_token = kl_token * mask
    return kl_token.sum(-1)  # (B,)


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
    rollouts = []
    device = get_device()
    prefix = torch.tensor([[BOS_ID]], device=device)
    corrects = []
    for _ in range(n_rollouts):
        outputs, _ = policy.generate(prefix)
        corrects.append(expression_reward(outputs[0].cpu().numpy().tolist()))
        # ToDo: pad outputs to ensure length is `MAX_LEN`
        paddings = torch.tensor(
            [[PAD_ID] * (MAX_LEN - outputs.shape[-1])],
            device=device,
            dtype=outputs.dtype,
        )

        outputs = torch.cat([outputs, paddings], dim=-1)
        rollouts.append(outputs)

    rollouts = torch.cat(rollouts, dim=0).to(dtype=torch.long)
    weights = torch.zeros(rollouts.shape, dtype=torch.float, device=device)
    weights[rollouts != PAD_ID] = 1.0
    r_rm = reward_model(rollouts).detach()  # Shape (B,)

    kl_pen = compute_kl_penalty(
        policy, ref_policy, rollouts, weights
    ).detach()  # Shape (B,)
    r_total = (r_rm - kl_beta * kl_pen).unsqueeze(-1)  # shape (B, 1)

    optimizer.zero_grad()
    # poor-man's advantages: subtract mean of the batch
    advantages = (r_total - r_total.mean()) / (r_total.std() + 1e-8)
    with torch.no_grad():
        old_logp = token_logprobs(policy, rollouts)
    new_logp = token_logprobs(policy, rollouts)

    ## PPO_LOSS
    policy_ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clip(policy_ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = torch.minimum(advantages * policy_ratio, advantages * clipped_ratio)
    # ignore updates to PAD tokens and take mean
    policy_loss = -torch.sum(policy_loss * weights) / torch.sum(weights)
    policy_loss.backward()
    optimizer.step()

    metrics_dict = {
        "mean_reward": torch.mean(r_rm).item(),
        "mean_kl": kl_pen.mean().item(),
        "policy_loss": policy_loss.item(),
        "mean_correct": torch.tensor(corrects).mean().item(),
    }
    return metrics_dict


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

    print("Stage 1: SFT pretraining...")
    ref_policy = pretrain_sft(n_steps=1000)

    print("\nStage 2: Reward model training...")
    reward_model = train_reward_model(n_pairs=1000)

    rm_score, kl, accuracy = [], [], []

    for beta in kl_betas:
        print(f"\nStage 3: RLHF fine-tuning with {beta=}...")
        # policy = type(ref_policy)()  # fresh policy same architecture
        device = get_device()
        policy = LanguageModel().to(device=device)
        policy.load_state_dict(ref_policy.state_dict())  # init from SFT
        optimizer = optim.Adam(policy.parameters(), lr=1e-4)

        logs = []
        rm_run, kl_run, acc_run = [], [], []
        for step in range(1000):
            log = rlhf_ppo_step(
                policy, ref_policy, reward_model, optimizer, kl_beta=beta, n_rollouts=64
            )
            rm_run.append(log["mean_reward"])
            kl_run.append(log["mean_kl"])
            acc_run.append(log["mean_correct"])
            logs.append(log)
            if step % 50 == 0:
                print(
                    f"Step {step}: reward={log['mean_reward']:.3f} "
                    f"kl={log['mean_kl']:.3f} acc={log['mean_correct']:.3f}"
                )
        rm_score.append(rm_run)
        kl.append(kl_run)
        accuracy.append(acc_run)

    # Plot RM score, KL divergence, and ground-truth accuracy side by side,
    # one line per β so reward hacking (RM score up, accuracy down) is visible.
    fig, (ax_rm, ax_kl, ax_acc) = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        (ax_rm, rm_score, "Reward model score"),
        (ax_kl, kl, "KL from reference policy"),
        (ax_acc, accuracy, "Ground-truth accuracy"),
    ]
    for ax, scores, title in panels:
        for beta, run in zip(kl_betas, scores):
            ax.plot(run, label=f"β={beta}")
        ax.set_title(title)
        ax.set_xlabel("PPO step")
        ax.legend()

    fig.tight_layout()
    fig.savefig("reward_hacking_experiment.png", dpi=150)
    print("\nSaved plot to reward_hacking_experiment.png")


if __name__ == "__main__":
    # print("Stage 1: SFT pretraining...")
    # ref_policy = pretrain_sft(n_steps=1000)

    # print("\nStage 2: Reward model training...")
    # reward_model = train_reward_model(n_pairs=3000)

    # print("\nStage 3: RLHF fine-tuning...")
    # # policy = type(ref_policy)()  # fresh policy same architecture
    # device = get_device()
    # policy = LanguageModel().to(device=device)
    # policy.load_state_dict(ref_policy.state_dict())  # init from SFT
    # optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # logs = []
    # for step in range(500):
    #     log = rlhf_ppo_step(
    #         policy, ref_policy, reward_model, optimizer, kl_beta=0.1, n_rollouts=64
    #     )
    #     logs.append(log)
    #     if step % 50 == 0:
    #         print(
    #             f"Step {step}: reward={log['mean_reward']:.3f} "
    #             f"kl={log['mean_kl']:.3f} acc={log['mean_correct']:.3f}"
    #         )

    print("\nReward Hacking Experiment...")
    reward_hacking_experiment()
