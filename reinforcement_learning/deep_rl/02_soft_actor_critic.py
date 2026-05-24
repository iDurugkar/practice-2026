"""
Assignment 2: Soft Actor-Critic — Maximum Entropy RL
=====================================================
SAC is the canonical maximum-entropy RL algorithm and the theoretical
foundation for much of modern LLM alignment (KL-regularized RL ≡ MaxEnt RL
with a KL penalty toward a reference policy — exactly what RLHF does).

Topics:
  - MaxEnt RL objective: J = E[Σ r_t + α H(π(·|s_t))]
  - Dual problem: automatic entropy tuning via constrained optimization
  - Twin Q-networks (TD3 trick) to reduce overestimation bias
  - Target entropy: H_target = -|A| (heuristic from Haarnoja et al.)
  - Off-policy: replay buffer, separate behavior and target networks
  - Connection to RLHF: SAC with reference policy = RLHF objective

Paper refs:
  - SAC (Haarnoja et al. 2018)  https://arxiv.org/abs/1801.01290
  - SAC Applications (Haarnoja et al. 2019)  https://arxiv.org/abs/1812.11103
  - KL-regularized RL ↔ MaxEnt (Ziebart 2010, Levine 2018)

Setup: pip install torch gymnasium numpy matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
from torch import Tensor
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Part 1: Replay buffer (off-policy backbone)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        # TODO: Use numpy arrays for efficient storage.
        # Pre-allocate arrays for: obs, next_obs, actions, rewards, dones.
        raise NotImplementedError

    def add(self, obs, action, reward, next_obs, done):
        """TODO: Circular buffer insertion."""
        raise NotImplementedError

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """TODO: Sample uniformly, return dict of tensors on CPU."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Networks
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Q(s,a) → scalar. Used twice (twin Q) to reduce overestimation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        # TODO: MLP: [obs, action] → hidden → hidden → 1
        raise NotImplementedError

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError


class GaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy: a = tanh(μ + σ ε), ε ~ N(0,I).
    The tanh squashing requires a log-det Jacobian correction to log_prob.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 action_scale: float = 1.0):
        super().__init__()
        self.action_scale = action_scale
        # TODO: MLP trunk → two heads (mean, log_std)
        raise NotImplementedError

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return (action, log_prob) where:
          x_t ~ N(μ, σ²I)
          action = tanh(x_t) * action_scale
          log_prob = log N(x_t; μ, σ) - Σ_d log(1 - tanh(x_t_d)²) - log(action_scale)

        The log-det correction: log|∂tanh(x)/∂x| = log(1 - tanh(x)²).
        Use numerically stable form: 2*(log 2 - x - softplus(-2x)).

        TODO: Implement. This is the most error-prone part of SAC.
        """
        raise NotImplementedError

    def get_action(self, obs: Tensor) -> Tensor:
        """TODO: Return deterministic action (tanh(mean)) for evaluation."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: Automatic entropy tuning
# ---------------------------------------------------------------------------
# SAC's temperature α is tuned to satisfy H(π) ≥ H_target = -|A|.
# This is solved as a Lagrangian dual problem:
#   α* = argmin_α E[-α log π_θ(a|s) - α H_target]
# α is learned via gradient descent on its own optimizer.

def make_entropy_tuner(target_entropy: float, init_alpha: float = 0.2
                       ) -> tuple[nn.Parameter, optim.Optimizer]:
    """
    TODO: Create log_alpha parameter and its Adam optimizer.
    Return (log_alpha, alpha_optimizer).
    Note: optimize log_alpha (not alpha) for positivity constraint.
    """
    raise NotImplementedError


def alpha_loss(log_alpha: nn.Parameter, log_probs: Tensor,
               target_entropy: float) -> Tensor:
    """
    L_α = -E[α (log π(a|s) + H_target)]
    TODO: Implement. Detach log_probs from policy gradient.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: SAC update step
# ---------------------------------------------------------------------------

def sac_update(
    policy: GaussianPolicy,
    q1: QNetwork, q2: QNetwork,
    q1_target: QNetwork, q2_target: QNetwork,
    policy_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    log_alpha: nn.Parameter,
    alpha_optimizer: optim.Optimizer,
    batch: dict[str, Tensor],
    gamma: float = 0.99,
    tau: float = 0.005,
    target_entropy: float = -1.0,
) -> dict:
    """
    Full SAC update:

    1. Q-targets: y = r + γ(1-d) [min(Q1', Q2')(s', a') - α log π(a'|s')]
    2. Q-loss: MSE(Q1(s,a), y) + MSE(Q2(s,a), y)    [twin Q]
    3. Policy loss: E[α log π(a|s) - min(Q1, Q2)(s, a)]
    4. Alpha loss: -E[α (log π(a|s) + H_target)]
    5. Soft target update: θ' ← τθ + (1-τ)θ'

    TODO: Implement all 5 steps. Return dict with losses and alpha value.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5: Training and analysis
# ---------------------------------------------------------------------------

class SACTrainer:
    def __init__(self, env_id: str = "HalfCheetah-v4",
                 buffer_capacity: int = 1_000_000,
                 batch_size: int = 256,
                 learning_starts: int = 5000,
                 gamma: float = 0.99,
                 tau: float = 0.005):
        self.env = gym.make(env_id)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        action_scale = float(self.env.action_space.high[0])
        target_entropy = -act_dim  # heuristic: H* = -|A|

        self.policy = GaussianPolicy(obs_dim, act_dim, action_scale=action_scale)
        self.q1 = QNetwork(obs_dim, act_dim)
        self.q2 = QNetwork(obs_dim, act_dim)
        self.q1_target = QNetwork(obs_dim, act_dim)
        self.q2_target = QNetwork(obs_dim, act_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.log_alpha, self.alpha_optimizer = make_entropy_tuner(target_entropy)
        self.target_entropy = target_entropy
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, act_dim)
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.tau = tau

    def train(self, total_timesteps: int = 1_000_000) -> list[dict]:
        """
        TODO: Collect one step per env step, update once per step after learning_starts.
        Log every 1000 steps: mean_return (from eval episodes), alpha, q_loss, policy_loss.
        Evaluate with 10 deterministic episodes (using get_action).
        """
        raise NotImplementedError


def compare_sac_vs_ppo():
    """
    TODO: Train SAC and PPO (from assignment 01) on HalfCheetah-v4 for 1M steps.
    Plot:
      (a) Return vs. timesteps
      (b) Sample efficiency (return vs. number of gradient updates)
      (c) Alpha (temperature) over training — what does convergence indicate?

    Expected: SAC significantly outperforms PPO on continuous control.
    """
    raise NotImplementedError


def kl_regularized_rl_connection():
    """
    TODO (Theory Exercise — implement or analyze):

    The RLHF objective for an LLM is:
      J(π) = E_{x~D, y~π}[r(x,y)] - β KL(π || π_ref)

    The MaxEnt RL objective is:
      J(π) = E[Σ r_t + α H(π(·|s_t))]
           = E[Σ r_t - α KL(π || Uniform)]

    Show that with π_ref = reference policy and uniform as the "prior":
      - RLHF objective ≡ MaxEnt RL objective with KL toward π_ref
      - The optimal closed-form solution is: π*(a|s) ∝ π_ref(a|s) exp(Q*(s,a)/β)
      - This is the "soft" Q-function solution used in RLHF

    Implement: given a toy bandit with π_ref and r(a), compute π* analytically
    and compare against SAC converged solution.
    """
    raise NotImplementedError


if __name__ == "__main__":
    trainer = SACTrainer(env_id="HalfCheetah-v4")
    logs = trainer.train(total_timesteps=500_000)

    returns = [l.get("mean_return", 0) for l in logs if "mean_return" in l]
    plt.plot(returns)
    plt.xlabel("Eval checkpoint")
    plt.ylabel("Mean Return")
    plt.title("SAC on HalfCheetah-v4")
    plt.savefig("sac_halfcheetah.png", dpi=150)
