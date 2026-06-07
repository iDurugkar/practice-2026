"""Sanity checks that the environment is wired up correctly."""

import torch

from utils import get_device, set_seed


def test_torch_tensor_math():
    x = torch.arange(6.0).reshape(2, 3)
    assert (x @ x.T).shape == (2, 2)


def test_get_device_runs_a_tensor():
    device = get_device()
    x = torch.ones(3, device=device)
    assert x.sum().item() == 3.0


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)
