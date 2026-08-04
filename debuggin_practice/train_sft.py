"""
Exercise 1 — instruction-tune SmolLM2-135M on a small synthetic task.

    python train_sft.py

Runs in a couple of minutes on a 4060 Ti. There are exactly two independent
bugs. Everything else in this file is intentional and correct.

Do not fix anything until you can state, in one sentence each, what the two
bugs are and which observed symptom each one produces.
"""

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 96
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 3
SEED = 0

VERBS = ["Reverse", "Uppercase", "Count the letters in"]
WORDS = ["orbit", "quasar", "lattice", "tundra", "cipher", "meadow", "photon", "kelp"]


def make_data(n=256):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        verb, word = rng.choice(VERBS), rng.choice(WORDS)
        if verb == "Reverse":
            resp = word[::-1]
        elif verb == "Uppercase":
            resp = word.upper()
        else:
            resp = str(len(word))
        rows.append({"instruction": f"{verb} the word '{word}'.", "response": resp})
    return rows


def format_prompt(instruction):
    return f"### User:\n{instruction}\n\n### Assistant:\n"


class SFTDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        prompt = format_prompt(row["instruction"])
        full = prompt + row["response"] + self.tok.eos_token

        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        full_ids = self.tok(full, add_special_tokens=False).input_ids[:MAX_LEN]

        return {"input_ids": full_ids, "prompt_len": len(prompt_ids)}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)

    input_ids, attn = [], []
    for b in batch:
        ids = b["input_ids"]
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)

    input_ids = torch.tensor(input_ids)
    attn = torch.tensor(attn)

    labels = input_ids.clone()
    # supervise the assistant's tokens only; padding never contributes
    labels[attn == 0] = -100
    for i, b in enumerate(batch):
        labels[i, :b['prompt_len']] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "prompt_lens": torch.tensor([b["prompt_len"] for b in batch]),
    }


def compute_loss(model, batch):
    # hand-rolled instead of model(..., labels=...) so per-token weighting
    # can be added later
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    # skip first logit
    logits = out.logits[:, :-1, :]
    # print(logits.shape)
    labels = batch["labels"][:, 1:]
    # print(labels.shape)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def sample(model, tok, instruction, max_new_tokens=24):
    model.eval()
    ids = tok(format_prompt(instruction), return_tensors="pt").to(DEVICE)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    model.train()
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(DEVICE)
    model.train()

    ds = SFTDataset(make_data(), tok)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    step = 0
    for epoch in range(EPOCHS):
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = compute_loss(model, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 20 == 0:
                print(f"epoch {epoch}  step {step:4d}  loss {loss.item():.4f}")
            step += 1

    print("\n--- greedy samples ---")
    for instruction in [
        "Reverse the word 'kelp'.",
        "Uppercase the word 'photon'.",
        "Count the letters in the word 'lattice'.",
    ]:
        print(repr(sample(model, tok, instruction)))
        print()


if __name__ == "__main__":
    main()
