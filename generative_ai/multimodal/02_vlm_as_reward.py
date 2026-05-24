"""
Assignment 2: Vision-Language Models as Reward Signals
=======================================================
A frontier topic: using VLMs (CLIP, GPT-4V, LLaVA) to provide dense reward
signals for embodied RL agents, replacing hand-crafted reward functions.

This directly connects multimodal GenAI to your RL expertise:
  - CLIP reward: r(s) = cosim(CLIP_img(s), CLIP_txt(goal_description))
  - VLM-as-judge: ask the model "has the agent achieved X?" given a screenshot
  - Dense shaping via visual goal representations

Topics:
  - CLIP-based reward shaping for goal-conditioned RL
  • Reward normalization and stability under VLM reward
  - Prompt sensitivity analysis (the VLM reward is a function of your prompt)
  - Reward hacking via adversarial image generation
  - Comparing CLIP reward vs. ground-truth sparse reward in a toy MDP

Paper refs:
  - CLIP-reward (Kwon et al. 2023)  https://arxiv.org/abs/2307.01927
  - VLP (Rocamonde et al. 2023)  https://arxiv.org/abs/2310.12921
  - RL-VLM-F (Rocamonde et al. 2023)  https://arxiv.org/abs/2307.07507

Setup: pip install torch torchvision transformers pillow gymnasium matplotlib
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Part 1: CLIP reward function
# ---------------------------------------------------------------------------

class CLIPReward:
    """
    Wraps CLIP to produce a scalar reward from (image, goal_text) pairs.
    r = cosine_similarity(CLIP_img(obs), CLIP_txt(goal)) ∈ [-1, 1]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def text_embedding(self, text: str) -> Tensor:
        """TODO: Return L2-normalized CLIP text embedding for `text`."""
        raise NotImplementedError

    @torch.no_grad()
    def image_embedding(self, image: Image.Image) -> Tensor:
        """TODO: Return L2-normalized CLIP image embedding for `image`."""
        raise NotImplementedError

    def reward(self, image: Image.Image, goal_text: str) -> float:
        """TODO: Return cosine similarity between image and goal embeddings."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Prompt sensitivity analysis
# ---------------------------------------------------------------------------
# The CLIP reward is a function of your prompt. Different phrasings of the
# same goal can give wildly different reward shapes — a major source of
# instability in practice.

def prompt_sensitivity_experiment(reward_fn: CLIPReward, test_images: list[Image.Image]):
    """
    For a fixed set of test images, compute CLIP reward under many prompt variants
    for the same semantic goal (e.g., "a robot arm holding a red block").

    Prompt variants to try:
      - "a robot arm holding a red block"
      - "robot grasping red cube"
      - "successful grasp"
      - "red block lifted"
      - "task completed"
      - "a photo of a robot arm with a red block"

    TODO:
      1. Compute rewards for each image × prompt combination.
      2. Plot a heatmap (images × prompts).
      3. Compute: rank correlation across prompts — do they agree on ordering?
      4. Report: which phrasings are most sensitive to image content?
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: CLIP reward in a toy goal-conditioned MDP
# ---------------------------------------------------------------------------
# Use a simple 2D navigation GridWorld where the "observation" is a rendered
# image and the goal is specified as text ("agent at top-right corner").
# Compare:
#   (a) Ground-truth sparse reward: +1 at goal cell, 0 elsewhere
#   (b) CLIP reward: cosim(render(state), goal_text)
#   (c) CLIP reward + running mean/std normalization

class GridWorldEnv(gym.Env):
    """5×5 grid. Agent starts at (0,0), renders to PIL Image for CLIP."""

    SIZE = 5

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(4)  # NESW
        self._pos = np.array([0, 0])

    def reset(self, **kwargs):
        self._pos = np.array([0, 0])
        return self._render(), {}

    def step(self, action: int):
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N E S W
        self._pos = np.clip(self._pos + deltas[action], 0, self.SIZE - 1)
        obs = self._render()
        sparse_reward = float(np.all(self._pos == [self.SIZE - 1, self.SIZE - 1]))
        return obs, sparse_reward, sparse_reward > 0, False, {}

    def _render(self) -> np.ndarray:
        """TODO: Render a 64×64 RGB image: white grid, blue agent, green goal."""
        raise NotImplementedError

    def render_pil(self) -> Image.Image:
        return Image.fromarray(self._render())


class RunningNormalizer:
    """Online mean/std normalization for reward stabilization."""

    def __init__(self, clip: float = 5.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip

    def update_and_normalize(self, r: float) -> float:
        """
        Welford online update for mean/variance.
        TODO: Update self.mean, self.var, self.count, then return
          clipped((r - mean) / (sqrt(var) + 1e-8), -clip, clip)
        """
        raise NotImplementedError


def compare_reward_signals(n_episodes: int = 200):
    """
    Train a tabular Q-agent on GridWorld using each of the three reward types.
    Plot learning curves (steps to goal vs. episode) for all three.

    TODO:
      - Tabular Q-learning with ε-greedy (ε annealed from 1.0 to 0.05)
      - γ = 0.99, α = 0.1
      - For CLIP reward: call reward_fn.reward(env.render_pil(), goal_text) at each step
      - Report: does CLIP reward recover the right policy?
        Does normalization help? What reward hacking do you observe?
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Reward hacking via adversarial images
# ---------------------------------------------------------------------------

def find_adversarial_image(
    reward_fn: CLIPReward,
    goal_text: str,
    n_steps: int = 500,
    lr: float = 0.01,
) -> Image.Image:
    """
    Maximize CLIP reward by gradient ascent on pixel values (no semantic constraint).
    This demonstrates reward hacking: the adversarial image achieves high CLIP
    reward without semantically achieving the goal.

    TODO:
      1. Initialize x ~ Uniform(0,1) of shape (1,3,224,224).
      2. Optimize x via Adam to maximize reward_fn.reward(x, goal_text).
      3. Plot: original random image → optimized image, and reward over steps.
      4. Discuss: what does this imply for using CLIP reward in RL?
    """
    raise NotImplementedError


if __name__ == "__main__":
    reward_fn = CLIPReward()

    print("=== Reward Hacking Demo ===")
    adv_img = find_adversarial_image(reward_fn, "robot arm successfully grasping red block")

    print("\n=== GridWorld Reward Signal Comparison ===")
    compare_reward_signals(n_episodes=300)
