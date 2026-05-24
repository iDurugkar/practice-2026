"""
Assignment 1: PPO — Implementation Details That Actually Matter
===============================================================
PPO is simple to describe and easy to get wrong. The implementation tricks
are what separate a working from a broken PPO. This assignment makes you
implement each piece from scratch and understand *why* each detail exists.

Topics (the "37 implementation details" distilled):
  - GAE-λ: bias-variance tradeoff in advantage estimation
  - Value function clipping (often omitted, sometimes harmful)
  - Gradient clipping vs. trust-region clipping
  - Observation and reward normalization (running stats, not batch)
  - Orthogonal initialization and tanh activations for stability
  - Mini-batch shuffling within an epoch
  - KL divergence monitoring as an early stopping signal
  - Entropy bonus: connection to maximum entropy RL (MaxEnt / SAC)

Paper refs:
  - PPO (Schulman et al. 2017)  https://arxiv.org/abs/1707.06347
  - Implementation matters (Engstrom et al. 2020)  https://arxiv.org/abs/2005.12729
  - 37 details (Huang et al. 2022)  https://arxiv.org/abs/2205.09123

Setup: pip install torch gymnasium numpy matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import Tensor
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Part 1: Network architecture with orthogonal init
# ---------------------------------------------------------------------------

def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """
    Orthogonal initialization. Engstrom et al. show this matters for
    on-policy methods — it preserves gradient norms at initialization.

    TODO: Apply nn.init.orthogonal_ with the given gain to layer.weight,
    and nn.init.constant_ 0 to layer.bias. Return layer.
    """
    raise NotImplementedError


class ActorCritic(nn.Module):
    """
    Shared-trunk MLP for continuous control. Separate heads for policy and value.
    Policy outputs Gaussian mean; log_std is a separate learnable parameter.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        # TODO: Build:
        #   trunk: [Linear(obs_dim, hidden), Tanh, Linear(hidden, hidden), Tanh]
        #   policy_head: Linear(hidden, act_dim) — init gain=0.01 (small for near-uniform init)
        #   value_head: Linear(hidden, 1) — init gain=1.0
        #   log_std: nn.Parameter of shape (act_dim,), init to 0
        # All trunk layers: gain=√2.
        raise NotImplementedError

    def forward(self, obs: Tensor) -> tuple[Normal, Tensor]:
        """Return (action_distribution, value_estimate)."""
        raise NotImplementedError

    def get_action(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample action, return (action, log_prob, value)."""
        raise NotImplementedError

    def evaluate_actions(self, obs: Tensor, actions: Tensor
                         ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (log_prob, entropy, value) for given obs-action pairs."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Running normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """
    Online Welford algorithm for obs/reward normalization.
    Critical: normalize observations for stable policy gradients.
    """

    def __init__(self, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        """TODO: Welford online update."""
        raise NotImplementedError

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """TODO: Return (x - mean) / sqrt(var + 1e-8), clipped."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: GAE-λ advantage estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: Tensor, values: Tensor, dones: Tensor,
    next_value: Tensor, gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    """
    Generalized Advantage Estimation (Schulman et al. 2016):
      δ_t = r_t + γ V(s_{t+1}) - V(s_t)
      Â_t = Σ_{k=0}^{T-t-1} (γλ)^k δ_{t+k}

    Computed in reverse for efficiency. dones masks episode boundaries.

    Returns: (advantages, returns) where returns = advantages + values.

    TODO: Implement. Note: at episode boundary, V(s_{t+1}) = 0 (no bootstrap).
    Also: returns (targets for value head) = advantages + values.detach().

    Key tradeoff: λ=1 → MC returns (low bias, high variance);
                  λ=0 → TD(0) (high bias, low variance).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    obs: Tensor, actions: Tensor, log_probs_old: Tensor,
    advantages: Tensor, returns: Tensor,
    clip_eps: float = 0.2,
    value_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    n_epochs: int = 10,
    n_minibatches: int = 4,
) -> dict:
    """
    PPO update over `n_epochs` epochs with mini-batch shuffling.

    Clipped objective: L = E[min(r_t Â, clip(r_t, 1-ε, 1+ε) Â)]
    Value loss (optionally clipped): L_V = max(MSE(V, R), clipped_MSE)
    Total: L_total = -L + c1 L_V - c2 H

    TODO: Implement full update loop. Normalize advantages within each minibatch.
    Return dict with mean: policy_loss, value_loss, entropy, approx_kl, clip_fraction.

    approx_kl ≈ mean((log_prob_new - log_prob_old)**2) / 2  (fast approximation)
    clip_fraction = mean(|r_t - 1| > ε)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5: Training loop with monitoring
# ---------------------------------------------------------------------------

class PPOTrainer:
    def __init__(self, env_id: str = "Hopper-v4", n_steps: int = 2048,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.env = gym.make(env_id)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))
        self.ret_norm = RunningMeanStd()
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def collect_rollout(self) -> dict:
        """
        TODO: Collect n_steps of experience. Normalize observations using obs_norm.
        Return dict with keys: obs, actions, log_probs, rewards, dones, values.
        """
        raise NotImplementedError

    def train(self, total_timesteps: int = 1_000_000) -> list[dict]:
        """
        TODO: Main training loop. At each iteration:
          1. collect_rollout()
          2. compute_gae()
          3. ppo_update()
          4. Log: mean_return, policy_loss, value_loss, entropy, approx_kl
          5. Early stop epoch if approx_kl > 0.02 (prevents too-large updates)

        Return list of log dicts per iteration.
        """
        raise NotImplementedError


def ablation_study():
    """
    TODO: Train PPO on Hopper-v4 with different configurations:
      - Full PPO (all tricks)
      - No GAE (λ=0, i.e., TD(0) advantages)
      - No observation normalization
      - No entropy bonus
      - No value clipping

    Plot learning curves side-by-side. Which tricks matter most?
    This reproduces (a simplified version of) Engstrom et al. 2020.
    """
    raise NotImplementedError


if __name__ == "__main__":
    trainer = PPOTrainer(env_id="Hopper-v4")
    logs = trainer.train(total_timesteps=500_000)

    returns = [l["mean_return"] for l in logs]
    plt.plot(returns)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Episode Return")
    plt.title("PPO on Hopper-v4")
    plt.savefig("ppo_hopper.png", dpi=150)
    print(f"Final return: {returns[-1]:.1f}")
