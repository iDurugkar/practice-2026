"""
Assignment 1: Tabular Methods — Convergence Theory & Implementation Nuances
===========================================================================
You know the algorithms. This assignment focuses on what's often glossed over:
  - Exact convergence conditions and rates
  - The role of stochastic approximation (Robbins-Monro) in Q-learning
  - Comparison: synchronous DP vs. asynchronous TD vs. model-based planning
  - Exploration: the difference between PAC-MDP, regret, and sample complexity

Topics:
  - Bellman contraction: ‖T^π v - T^π u‖_∞ ≤ γ ‖v - u‖_∞ → proof in code
  - Convergence rate of policy iteration vs. value iteration
  - SARSA vs Q-learning: on-policy vs off-policy convergence theorems
  - Rmax and the exploration bonus perspective
  - Dyna: integrating a learned model with TD updates

Paper refs:
  - Watkins & Dayan 1992 Q-learning convergence proof
  - Brafman & Tennenholtz 2002 Rmax  https://jmlr.org/papers/v3/brafman02a.html
  - Dyna (Sutton 1991)

Setup: pip install numpy gymnasium matplotlib scipy
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Environment: Stochastic GridWorld (demonstrates non-trivial convergence)
# ---------------------------------------------------------------------------

class StochasticGridWorld:
    """
    5×5 grid, goal at (4,4), cliff row at row 3.
    Transition: with prob 0.2, wind pushes agent in a random direction.
    This tests convergence under stochastic dynamics.
    """

    SIZE = 5
    GOAL = (4, 4)
    CLIFF = {(3, c) for c in range(1, 4)}  # r=-10 and back to start
    WIND_PROB = 0.2

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state = (0, 0)

    @property
    def n_states(self) -> int:
        return self.SIZE * self.SIZE

    @property
    def n_actions(self) -> int:
        return 4  # N E S W

    def state_idx(self, s: tuple) -> int:
        return s[0] * self.SIZE + s[1]

    def reset(self) -> tuple:
        self.state = (0, 0)
        return self.state

    def step(self, action: int) -> tuple[tuple, float, bool]:
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        if self.rng.random() < self.WIND_PROB:
            action = self.rng.integers(4)
        r, c = self.state
        dr, dc = deltas[action]
        ns = (np.clip(r + dr, 0, self.SIZE-1), np.clip(c + dc, 0, self.SIZE-1))
        if ns in self.CLIFF:
            return (0, 0), -10.0, False
        if ns == self.GOAL:
            return ns, 1.0, True
        self.state = ns
        return ns, -0.01, False

    def get_transition_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build exact P[s,a,s'] and R[s,a] for DP baselines.
        TODO: Enumerate all (s,a) → weighted s' outcomes including wind.
        Return P shape (S, A, S') and R shape (S, A).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 1: Bellman contraction — verify empirically
# ---------------------------------------------------------------------------

def bellman_backup(V: np.ndarray, P: np.ndarray, R: np.ndarray,
                   gamma: float, pi: np.ndarray) -> np.ndarray:
    """
    Synchronous Bellman backup for a fixed policy π:
      V'(s) = R(s, π(s)) + γ Σ_s' P(s, π(s), s') V(s')

    TODO: Implement vectorized. Return V' of shape (S,).
    """
    raise NotImplementedError


def verify_contraction(env: StochasticGridWorld, gamma: float = 0.99):
    """
    TODO: For a random policy, run 20 Bellman backups starting from two
    different initial value functions V1, V2.
    At each step, compute ‖V1_k - V2_k‖_∞ and verify it decreases by
    exactly γ each step (within floating-point tolerance).
    Plot convergence of ‖V1_k - V2_k‖_∞ vs. γ^k ‖V1_0 - V2_0‖_∞.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Policy iteration vs. value iteration — convergence rate
# ---------------------------------------------------------------------------

def policy_evaluation(P, R, pi, gamma, tol=1e-10) -> np.ndarray:
    """
    Solve (I - γ P^π) V = R^π exactly via linear system.
    This is the "gold standard" used in policy iteration.
    TODO: Use np.linalg.solve. Return V of shape (S,).
    """
    raise NotImplementedError


def policy_iteration(P, R, gamma) -> tuple[np.ndarray, np.ndarray, list]:
    """
    TODO: Run policy iteration to convergence.
    Return (optimal_V, optimal_pi, iterations_to_convergence).
    Track number of policy evaluation calls and number of policy changes.
    """
    raise NotImplementedError


def value_iteration(P, R, gamma, tol=1e-8) -> tuple[np.ndarray, np.ndarray, list]:
    """
    TODO: Run value iteration.
    Return (optimal_V, optimal_pi, list_of_bellman_errors_per_iter).
    """
    raise NotImplementedError


def compare_convergence():
    """
    TODO: Plot ‖V_k - V*‖_∞ vs. iteration for both methods on same axes.
    Note: PI converges in O(log(1/(1-γ)) / log(1/γ)) iterations,
    VI in O(log(1/ε) / log(1/γ)) — compute both theoretical bounds and overlay.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: SARSA vs Q-learning — off-policy divergence near the cliff
# ---------------------------------------------------------------------------

def q_learning(env: StochasticGridWorld, n_episodes: int, gamma: float,
               alpha_fn, eps_fn) -> tuple[np.ndarray, list]:
    """
    Q-learning with time-varying step size α_fn(t) and exploration eps_fn(t).
    For convergence: Σ α_t = ∞ and Σ α_t² < ∞ (Robbins-Monro conditions).

    TODO: Implement. Return (Q table, episode_returns).
    """
    raise NotImplementedError


def sarsa(env: StochasticGridWorld, n_episodes: int, gamma: float,
          alpha_fn, eps_fn) -> tuple[np.ndarray, list]:
    """
    On-policy SARSA. Same interface as q_learning.
    TODO: Implement. Key difference: next action is sampled from current policy,
    not greedy.
    """
    raise NotImplementedError


def compare_cliff_policies():
    """
    TODO: Train both agents on StochasticGridWorld.
    Compare learned paths: Q-learning should prefer the shortest path
    (greedy, near cliff); SARSA should learn a safer route.
    Compute and compare:
      - Mean episode return (online)
      - Greedy policy path length
      - Number of cliff falls per 1000 episodes
    This empirically demonstrates the on-policy safety advantage of SARSA.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Dyna — integrating model learning with TD
# ---------------------------------------------------------------------------

def dyna_q(env: StochasticGridWorld, n_episodes: int, n_planning_steps: int,
           gamma: float, alpha: float, eps: float) -> tuple[np.ndarray, list]:
    """
    Dyna-Q: after each real experience, do n_planning_steps of simulated
    experience using the learned model.

    Model: store observed (s,a) → (r, s') transitions (deterministic approximation).

    TODO: Implement. Show that Dyna-Q reaches near-optimal policy
    3-5x faster in wall-clock steps than Q-learning alone.
    """
    raise NotImplementedError


if __name__ == "__main__":
    env = StochasticGridWorld(seed=42)
    P, R = env.get_transition_matrix()

    print("=== Bellman Contraction Verification ===")
    verify_contraction(env)

    print("\n=== PI vs VI Convergence ===")
    compare_convergence()

    print("\n=== SARSA vs Q-learning on Cliff ===")
    compare_cliff_policies()
