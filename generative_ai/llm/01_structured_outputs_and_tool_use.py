"""
Assignment 1: Structured Generation & Tool Use
===============================================
Relevant to LLM-RL because: reliable structured output is what makes LLMs
usable as policy networks — the model must emit well-typed actions, not free text.

Topics:
  - Constrained decoding via prompt + output parsing
  - Tool/function calling as a typed action space
  - Multi-turn tool loops (ReAct-style)
  - Measuring reliability across temperatures

Paper refs:
  - ReAct (Yao et al. 2023)  https://arxiv.org/abs/2210.03629
  - Toolformer (Schick et al. 2023)  https://arxiv.org/abs/2302.04761

Setup: pip install anthropic && export ANTHROPIC_API_KEY=...
"""

import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Part 1: Reliable structured output via prompt constraints
# ---------------------------------------------------------------------------
# LLMs are unreliable emitters of JSON without guidance.
# Task: get the model to classify a research abstract into
#   {"topic": str, "methodology": str, "claims_strength": "weak"|"moderate"|"strong"}
# and measure parse failure rate across 20 calls at temperature 1.0.

ABSTRACT = """
We present a novel approach to offline RL that combines conservative Q-learning
with diffusion-based data augmentation. On D4RL benchmarks, our method achieves
state-of-the-art on hopper-medium-v2 (95.2) and halfcheetah-medium-v2 (79.1)
with only 100k gradient steps, a 3x reduction over CQL baselines.
"""

def classify_abstract_unstructured(abstract: str) -> dict | None:
    """Prompt without explicit format enforcement. Measure parse reliability."""
    # TODO: Craft a prompt that asks for JSON output.
    # Return parsed dict or None on parse failure.
    # Run 20 times and report failure rate.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Tool use as a typed action space
# ---------------------------------------------------------------------------
# Define a toy "calculator + search" tool set.
# The model must plan which tools to call, in what order, to answer a
# multi-step question — this is structurally identical to a discrete action MDP.

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Returns a float.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "lookup_paper_citation_count",
        "description": "Return the citation count for a paper by title (mocked).",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
]

MOCK_CITATIONS = {
    "Proximal Policy Optimization": 12000,
    "Soft Actor-Critic": 8500,
    "Decision Transformer": 3200,
}

def mock_tool_call(name: str, inputs: dict) -> str:
    if name == "calculator":
        return str(eval(inputs["expression"]))  # noqa: S307 — toy example
    if name == "lookup_paper_citation_count":
        count = MOCK_CITATIONS.get(inputs["title"], 0)
        return json.dumps({"citations": count})
    return "unknown tool"


def run_tool_loop(question: str) -> str:
    """
    Run an agentic tool loop until the model emits stop_reason='end_turn'.
    Return the final text answer.

    TODO: Implement the multi-turn loop:
      1. Send the question with TOOLS defined above.
      2. If stop_reason == "tool_use", execute each tool_use block via mock_tool_call.
      3. Feed results back as a "tool" role message.
      4. Repeat until stop_reason == "end_turn".
      5. Return the final assistant text.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: ReAct-style reasoning trace
# ---------------------------------------------------------------------------
# Prompt the model to emit Thought / Action / Observation traces before
# giving a final answer. Compare answer quality vs. direct answering on 5
# multi-step RL questions (e.g., "What is the KL divergence penalty weight
# used in the original RLHF paper by Ziegler et al.?").

RL_QUESTIONS = [
    "What discount factor γ did the original DQN paper use for Atari?",
    "In PPO, what is the typical clip ratio ε, and why was that value chosen?",
    "How many gradient steps per environment step does SAC typically use?",
    "What is the key difference between TD3 and DDPG in terms of variance reduction?",
    "In GRPO, how are the per-token advantages normalized within a group?",
]

def react_answer(question: str) -> str:
    """
    TODO: Prompt with a ReAct system prompt that enforces the
    Thought / Action / Observation / Answer format.
    For this exercise, Action can only be 'think' (no real tools needed).
    Return the final Answer portion.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Reliability experiment
# ---------------------------------------------------------------------------

def reliability_experiment():
    """
    TODO: For each question in RL_QUESTIONS, call both direct answering and
    react_answer. Use an LLM judge (a second call) to score correctness 1-5.
    Print a table comparing mean scores and response length.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Smoke test tool loop
    answer = run_tool_loop(
        "How many times more cited is PPO than Decision Transformer? "
        "Express as a ratio rounded to 1 decimal."
    )
    print("Tool loop answer:", answer)

    reliability_experiment()
