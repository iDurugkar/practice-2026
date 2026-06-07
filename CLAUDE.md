# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal study sandbox for practicing ML, GenAI, and RL by implementing
research-level concepts from scratch. It is **not** a library or application —
there is no package to build and nothing imports across the topic directories.

Each `.py` file under `pytorch_foundations/`, `generative_ai/`, and
`reinforcement_learning/` is a
**self-contained assignment**: a guided skeleton with theory, paper references,
and deliberately unimplemented stubs (`raise NotImplementedError`) that the user
fills in. The numeric prefixes (`01_`, `02_`) indicate intended order within a
subtopic.

When asked to "do" or "solve" an assignment, the task is to replace the
`raise NotImplementedError` / `TODO` bodies with correct implementations while
preserving the file's structure, signatures, and pedagogical comments — not to
rewrite the scaffolding.

## Environment & commands

Uses [uv](https://docs.astral.sh/uv/) (Python 3.12, PyTorch with MPS). The
installed env is **lean** — only `torch`, `torchvision`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib` (plus `scipy`/`pillow` transitively), and dev tools
(`jupyterlab`, `ipykernel`, `tqdm`, `pytest`).

```bash
uv sync                                    # install deps into .venv
uv run python <path/to/assignment.py>      # run one assignment (executes its __main__)
uv run pytest                              # run the test suite
uv run pytest test_setup.py::test_set_seed_is_deterministic   # single test
uv run jupyter lab                         # notebooks/
uv add <pkg>                               # add a dependency (--dev for tooling)
```

`uv run` auto-syncs first, so the venv rarely needs manual activation.

## Missing dependencies per assignment

Several assignments import libraries that are **not** in the lean env. Each
file's module docstring has a `Setup: pip install ...` line listing them. Before
running (or after implementing) such an assignment, install the extras with uv:

- RL (`classic/`, `deep_rl/`, multimodal VLM): `uv add gymnasium`
- LLM assignments (`generative_ai/llm/`): `uv add anthropic` (needs an API key)
- Multimodal VLM-as-reward: `uv add transformers`

If asked to keep the env lean, prefer implementing/verifying the pure-PyTorch
parts and note which `__main__` demos require the extra install.

## Assignment file anatomy

Every assignment follows the same convention — recognizing it makes edits faster:

- **Module docstring**: a Topics list and `Paper refs` with arXiv links. These
  define the intended algorithm; consult them when an implementation is ambiguous.
- **Parts**, separated by `# ----` banner comments (`Part 1`, `Part 2`, …),
  building from a toy/analytic case to the full method.
- **Stubs**: each `raise NotImplementedError` is preceded by a `TODO:` comment
  spelling out exactly what to compute (often with the governing equation).
- **`__main__` block**: a runnable demo or training loop that wires the
  implemented pieces together (e.g. PPO trains on `Hopper-v4` and saves a PNG).
  Use it to validate an implementation end-to-end.

## Shared helpers

`utils.py` (repo root) provides `get_device()` (cuda → mps → cpu) and
`set_seed()`. Assignments are currently standalone and don't import it; prefer
these helpers over re-deriving device/seed logic when adding harness code.

## ToDo
* Implement a forward backward pass
* Tensor manipulation examples
* implement SFT training loop
* implement on-policy distillation loop
