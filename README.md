# Generative AI & Reinforcement Learning Practice

A sandbox for experimenting with generative AI and reinforcement learning techniques.

## Structure

```
practice_2026/
├── generative_ai/          # GenAI experiments
│   ├── llm/                # LLM prompting, fine-tuning, RAG
│   ├── diffusion/          # Image/audio generation
│   └── multimodal/         # Vision-language models
├── reinforcement_learning/ # RL experiments
│   ├── classic/            # CartPole, MountainCar, etc.
│   ├── deep_rl/            # DQN, PPO, A3C
│   └── llm_rl/             # RLHF, GRPO, reward modeling
└── notebooks/              # Exploratory Jupyter notebooks
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
`pyproject.toml` / `uv.lock` (Python 3.12, PyTorch with MPS support).

```bash
uv sync                      # create .venv and install everything (incl. dev tools)
```

### Running things

```bash
uv run python script.py      # run a script in the env
uv run jupyter lab           # launch JupyterLab
uv run pytest                # run tests
```

`uv run <cmd>` auto-syncs the env first, so you rarely need to activate the venv
manually. To add a package: `uv add <pkg>` (or `uv add --dev <pkg>` for tooling).

### Quick start

```python
from utils import get_device
import torch

device = get_device()                  # -> "mps" on Apple Silicon
x = torch.randn(4, 4, device=device)
```
