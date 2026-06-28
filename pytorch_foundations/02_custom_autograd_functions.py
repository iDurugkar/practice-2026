"""
Assignment 2: Custom autograd Functions & the Straight-Through Estimator
========================================================================
Sometimes you need an op autograd doesn't provide, a numerically stable
backward, or a gradient *through* a non-differentiable operation. That's what
torch.autograd.Function is for. You'll write forward/backward by hand, verify
with gradcheck, and implement the straight-through estimator (STE) — the trick
behind quantized and discrete-latent networks.

Topics:
  - torch.autograd.Function: forward, backward, save_for_backward
  - Numerically stable LogSumExp and its gradient (the softmax)
  - torch.autograd.gradcheck: analytic vs numerical Jacobian in double precision
  - Straight-through estimator for non-differentiable ops (round / quantize)

Refs:
  - Estimating gradients through stochastic neurons (Bengio et al. 2013)  https://arxiv.org/abs/1308.3432
  - Extending PyTorch autograd  https://pytorch.org/docs/stable/notes/extending.html

Setup: pip install torch   (in the lean env)
"""

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.autograd import Function

# ---------------------------------------------------------------------------
# Part 1: A numerically stable LogSumExp Function
# ---------------------------------------------------------------------------


class LogSumExp(Function):
    """
    Differentiable logsumexp over the last dimension:
        out = log(sum_j exp(x_j))
    Stable via max-subtraction. Its gradient is softmax(x).
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # TODO:
        #   m   = x.max(dim=-1, keepdim=True).values
        #   z   = (x - m).exp()
        #   out = z.sum(-1, keepdim=True).log() + m      # then squeeze last dim
        #   ctx.save_for_backward(z / z.sum(-1, keepdim=True))   # = softmax(x)
        #   return out.squeeze(-1)                       # shape x.shape[:-1]
        m = x.max(dim=-1, keepdim=True).values
        z = (x - m).exp()
        out = z.sum(-1, keepdim=True).log() + m
        ctx.save_for_backward(z / z.sum(-1, keepdim=True))  # softmax
        return out.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # TODO: d/dx logsumexp(x) = softmax(x). grad_out has shape x.shape[:-1];
        # unsqueeze its last dim and multiply by the saved softmax.
        softmax = ctx.saved_tensors[0]
        # print(softmax.shape, ctx.saved_tensors)
        dx = grad_output.unsqueeze(-1) * softmax
        return dx


def logsumexp(x: Tensor) -> Tensor:
    return LogSumExp.apply(x)


# ---------------------------------------------------------------------------
# Part 2: Straight-Through Estimator
# ---------------------------------------------------------------------------


class RoundSTE(Function):
    """
    Forward: round to nearest integer (non-differentiable, zero gradient a.e.).
    Backward: pretend it was the identity — pass the gradient straight through.
    This is what lets you train through a hard quantization step.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # TODO: return torch.round(x). Nothing needs to be saved.
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        # TODO: identity backward — return grad_out unchanged.
        return grad_out


def round_ste(x: Tensor) -> Tensor:
    return RoundSTE.apply(x)


# ---------------------------------------------------------------------------
# Part 3: Train through a non-differentiable quantizer with the STE
# ---------------------------------------------------------------------------


def fit_with_quantizer(steps: int = 300, lr: float = 0.05):
    """
    Toy task: drive a scalar w so that round_ste(w) hits the target value 3,
    starting from w = 0. Without the STE the gradient is zero a.e. and w never
    moves; with it, w climbs toward 3. Return the loss history (list of floats).

    TODO:
      - w   = torch.zeros(1, requires_grad=True)
      - opt = torch.optim.SGD([w], lr=lr)
      - loop: opt.zero_grad(); loss = (round_ste(w) - 3.0).pow(2).mean();
        loss.backward(); opt.step(); record loss.item().
    """
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = (round_ste(w) - 3.0).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    torch.manual_seed(0)

    # Part 1: verify LogSumExp against torch and with gradcheck.
    x = torch.randn(4, 7, dtype=torch.double, requires_grad=True)
    print(
        "logsumexp max err vs torch:",
        (logsumexp(x) - torch.logsumexp(x, dim=-1)).abs().max().item(),
    )
    print("gradcheck LogSumExp:", torch.autograd.gradcheck(logsumexp, (x,)))

    # Part 2: STE forward rounds, backward is identity (ones).
    z = torch.tensor([0.2, 1.7, -0.4], requires_grad=True)
    round_ste(z).sum().backward()
    print("STE forward:", round_ste(z).tolist(), "| backward grad:", z.grad.tolist())

    # Part 3: training through the quantizer.
    hist = fit_with_quantizer()
    print(f"quantizer training final loss: {hist[-1]:.4f} (expect ~0)")
    plt.plot(hist)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig("autograd_testing.png")
    plt.cla()
