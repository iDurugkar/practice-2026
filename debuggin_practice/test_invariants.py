"""
Invariant harness — the stubs are yours to fill in.

    pytest test_invariants.py -x -q

Rule for this exercise: write the test BEFORE you look for the bug it is
meant to catch. If a test passes when you expected it to fail, that is
information too — write down what it rules out.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_sft import MODEL_NAME, SFTDataset, collate, compute_loss, make_data


def _fixture(n=4):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    rows = make_data(n)
    ds = SFTDataset(rows, tok)
    batch = collate([ds[i] for i in range(n)], tok.pad_token_id)
    return tok, model, ds, batch


def test_loss_at_init_is_not_uniform():
    """
    A pretrained model scoring in-distribution English should sit a few nats
    above zero, NOT at ln|V|.

    Compute ln(vocab_size). Compute the loss at init. Assert the gap.
    What does it mean if the loss lands within ~0.3 of ln|V|?
    """
    raise NotImplementedError


def test_loss_is_invariant_to_padding():
    """
    Build the same examples twice: once collated alone, once collated in a
    batch that forces a much longer pad width (append a long dummy example).

    The per-example loss must match to ~1e-5. If it does not, the mask is
    wrong somewhere. Note: you will need to reduce per-example rather than
    over the whole flattened batch to compare these.
    """
    raise NotImplementedError


def test_supervised_tokens_are_only_the_response():
    """
    No forward pass needed. For each row in the batch, decode exactly the
    positions where labels != -100 and assert the decoded string equals the
    response (plus its terminator) and nothing else.

    This is the cheapest test in the file and it catches the most.
    """
    raise NotImplementedError


def test_labels_predict_the_next_token():
    """
    Alignment check. For one unpadded example, assert that the label the loss
    holds logits[t] accountable for is input_ids[t+1], not input_ids[t].

    Hint: you cannot see this from `labels` alone -- the pairing is created
    inside compute_loss. Reason about which index of `labels` gets matched to
    which row of `logits` by F.cross_entropy after both are flattened.
    """
    raise NotImplementedError


def test_can_overfit_one_batch():
    """
    Train a single batch for ~200 steps at lr 1e-4. Loss should approach 0.

    Then the sharper question: if it DOES reach ~0 but generation is still
    broken, what class of bug does that point to, and what class does it
    rule out?
    """
    raise NotImplementedError
