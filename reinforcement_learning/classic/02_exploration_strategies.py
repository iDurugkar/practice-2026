"""
Assignment 2: Exploration — Regret Bounds, UCB, and Posterior Sampling
=======================================================================
Exploration is where the RL theory gets interesting. This assignment
covers the PAC-MDP vs. regret frameworks, and implements the three main
paradigms: optimism (UCB), posterior sampling (PSRL), and count-based bonuses.

Topics:
  - Multi-armed bandit as a testbed: regret vs. PAC-MDP distinctions
  - UCB1 and its O(√(KT log T)) regret bound — verify empirically
  - Thompson Sampling / Posterior Sampling RL (Osband et al.)
  - Count-based exploration bonuses: r+ = r + β/√n(s,a)
  - Intrinsic motivation: prediction error as exploration bonus (RND prototype)

Paper refs:
  - UCB1 (Auer et al. 2002)
  - PSRL (Strens 2000, Osband et al. 2013)  https://arxiv.org/abs/1306.0940
  - RND (Burda et al. 2018)  https://arxiv.org/abs/1810.12894
  - Count-based (Bellemare et al. 2016)  https://arxiv.org/abs/1606.01868

Setup: pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Part 1: Bandit testbed
# ---------------------------------------------------------------------------

class BernoulliBandit:
    """K-armed Bernoulli bandit with fixed unknown means μ_k."""

    def __init__(self, means: list[float], seed: int = 0):
        self.means = np.array(means)
        self.K = len(means)
        self.rng = np.random.default_rng(seed)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.means[arm])

    @property
    def optimal_mean(self) -> float:
        return self.means.max()

    def regret(self, arm: int) -> float:
        return self.optimal_mean - self.means[arm]


def compute_regret_curve(policy_fn: Callable, bandit: BernoulliBandit,
                         T: int, n_seeds: int = 20) -> np.ndarray:
    """
    Run `policy_fn` for T steps across n_seeds.
    policy_fn(counts, rewards, t) → arm_index.
    Return cumulative regret array of shape (n_seeds, T).
    """
    regrets = np.zeros((n_seeds, T))
    for seed in range(n_seeds):
        b = BernoulliBandit(bandit.means, seed=seed)
        counts = np.zeros(b.K)
        rewards = np.zeros(b.K)
        cum_regret = 0.0
        for t in range(T):
            arm = policy_fn(counts, rewards, t)
            r = b.pull(arm)
            counts[arm] += 1
            rewards[arm] += r
            cum_regret += b.regret(arm)
            regrets[seed, t] = cum_regret
    return regrets


# ---------------------------------------------------------------------------
# Part 2: UCB1 — optimism in the face of uncertainty
# ---------------------------------------------------------------------------

def ucb1(counts: np.ndarray, rewards: np.ndarray, t: int) -> int:
    """
    UCB1: arm = argmax_k [ μ̂_k + √(2 log(t+1) / n_k) ]
    Handle unplayed arms (pull each once first).

    TODO: Implement. The confidence bonus √(2 log t / n) comes from Hoeffding's
    inequality: P(|μ̂ - μ| > ε) ≤ 2 exp(-2 n ε²).
    """
    raise NotImplementedError


def verify_ucb1_regret_bound(T: int = 10000):
    """
    UCB1 achieves: E[R_T] ≤ Σ_k (8 log T / Δ_k) + (1 + π²/3) Σ_k Δ_k
    where Δ_k = μ* - μ_k.

    TODO:
      1. Run UCB1 on a 5-armed bandit for T=10000 steps.
      2. Compute the theoretical upper bound from the formula above.
      3. Plot: empirical regret (mean ± std), O(√(KT log T)) simple bound,
         and the gap-dependent bound.
      4. Comment on when gap-dependent bounds are tighter.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: Thompson Sampling — Bayesian posterior sampling
# ---------------------------------------------------------------------------

def thompson_sampling(counts: np.ndarray, rewards: np.ndarray, t: int) -> int:
    """
    Maintain Beta(α_k, β_k) posterior for each arm's mean.
    Sample θ_k ~ Beta(α_k, β_k), select argmax_k θ_k.

    Prior: Beta(1,1) = Uniform(0,1).
    Update: on reward r ∈ {0,1}: α_k += r, β_k += (1-r).

    TODO: counts[k] = n_k (total pulls), rewards[k] = Σ r_k (total rewards).
    """
    raise NotImplementedError


def compare_exploration_strategies(T: int = 5000):
    """
    Compare on a 10-armed bandit: UCB1, Thompson Sampling, ε-greedy (ε=0.1),
    greedy, and random.

    TODO:
      1. Run all strategies for T steps, n_seeds=50.
      2. Plot mean cumulative regret with 95% CI bands.
      3. Plot: fraction of optimal arm pulls per time step.
      4. Report: at T=5000, how does regret rank? Does Thompson beat UCB1?
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Count-based exploration in tabular MDP
# ---------------------------------------------------------------------------

def count_based_bonus(n_sa: int, beta: float = 0.1) -> float:
    """
    Intrinsic reward bonus: r+ = β / √(n(s,a) + 1)
    This is the MBIE-EB / count-based formulation.
    """
    return beta / np.sqrt(n_sa + 1)


def q_learning_with_bonus(env, n_episodes: int, gamma: float, alpha: float,
                           eps: float, beta: float) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Q-learning + count-based exploration bonus.
    r_augmented = r_extrinsic + count_based_bonus(n(s,a))

    TODO: Implement. Track visitation counts separately.
    Return (Q, episode_returns, visitation_counts[S, A]).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5: Learned curiosity — Random Network Distillation (prototype)
# ---------------------------------------------------------------------------
# RND uses prediction error of a fixed random network as an exploration bonus.
# Bonus is high for novel states, low for frequently visited ones.
# This is a continuous-state analogue of count-based exploration.

import torch
import torch.nn as nn
import torch.optim as optim

class RNDExploration:
    """
    Two networks: target (frozen random) and predictor (trained).
    Bonus: ||f_target(s) - f_predictor(s)||²

    High error → novel state → high bonus.
    As the predictor learns, bonus decreases for visited states.
    """

    def __init__(self, obs_dim: int, embed_dim: int = 64):
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim)
        )
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=3e-4)

    def bonus(self, obs: torch.Tensor) -> float:
        """TODO: Return ||target(obs) - predictor(obs)||² (no grad)."""
        raise NotImplementedError

    def update(self, obs: torch.Tensor) -> float:
        """TODO: Minimize predictor loss and return the loss value."""
        raise NotImplementedError


def rnd_experiment():
    """
    Apply RND to CartPole-v1 (continuous observations):
      1. Train a simple policy with Q-network + RND bonus.
      2. Track: intrinsic reward curve over training.
      3. Visualize: bonus magnitude in state space (sample s from env, plot heatmap).
      4. Compare: learning speed vs. pure ε-greedy exploration.

    TODO: Implement full training loop.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("=== UCB1 Regret Bound Verification ===")
    verify_ucb1_regret_bound(T=10000)

    print("\n=== Exploration Strategy Comparison ===")
    compare_exploration_strategies(T=5000)

    print("\n=== RND Curiosity Experiment ===")
    rnd_experiment()
