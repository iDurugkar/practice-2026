"""
Assignment 1: Tensor Mechanics & Autograd — The Bedrock
=======================================================
Before any model, you need fluency in the operations every model is built from:
broadcasting, einsum, and reverse-mode autodiff. This assignment drills the
mechanics and makes you derive a backward pass by hand so autograd stops being
magic.

Topics:
  - Broadcasting semantics and avoiding hidden O(n*m*d) memory blowups
  - einsum as one notation for matmul / batched / contraction ops
  - Reverse-mode autodiff: deriving and verifying an analytic backward pass
  - Gradient checking with finite differences

Refs:
  - The matrix calculus you need for deep learning (Parr & Howard)  https://arxiv.org/abs/1802.01528
  - PyTorch autograd mechanics  https://pytorch.org/docs/stable/notes/autograd.html

Setup: pip install torch numpy matplotlib   (all in the lean env)
"""

import matplotlib.pyplot as plt
import torch
from pandas.io.sql import table_exists
from torch import Tensor

# ---------------------------------------------------------------------------
# Part 1: Broadcasting & einsum — vectorize without loops
# ---------------------------------------------------------------------------


def pairwise_sq_distances(X: Tensor, Y: Tensor) -> Tensor:
    """
    Squared Euclidean distance between every row of X (n, d) and Y (m, d).
    Return a (n, m) matrix D with D[i, j] = ||X[i] - Y[j]||^2.

    TODO: Implement WITHOUT any Python loop and WITHOUT materializing the
    (n, m, d) difference tensor. Use the identity
        ||x - y||^2 = ||x||^2 - 2 x·y + ||y||^2
    i.e. row norms broadcast against a (n, m) inner-product matrix (X @ Y.T).
    """
    x_sq = torch.einsum("ik, ik -> i", X, X)
    y_sq = torch.einsum("jk, jk -> j", Y, Y)
    cross = torch.einsum("ik, jk -> ij", X, Y)
    return x_sq[:, None] - 2 * cross + y_sq[None, :]


def batched_bilinear(x: Tensor, A: Tensor, y: Tensor) -> Tensor:
    """
    Batched bilinear form. x: (b, p), A: (b, p, q), y: (b, q).
    Return shape (b,) where out[k] = x[k] @ A[k] @ y[k].

    TODO: Implement with a single torch.einsum call (no loop, no intermediate
    stacking). Figure out the right subscript string for "bp,bpq,bq->b".
    """
    return torch.einsum("bp, bpq, bq -> b", x, A, y)


def _reference_bilinear(x: Tensor, A: Tensor, y: Tensor) -> Tensor:
    return torch.stack([x[k] @ A[k] @ y[k] for k in range(x.shape[0])])


# ---------------------------------------------------------------------------
# Part 2: Derive a backward pass by hand
# ---------------------------------------------------------------------------
# Model: y = x @ W.T + b ; loss = mean over all n*d_out entries of (y - t)^2.
# Compute dL/dW, dL/db, dL/dx analytically and check against autograd + finite
# differences.


def linear_mse_forward(x: Tensor, W: Tensor, b: Tensor, t: Tensor) -> Tensor:
    """Forward pass returning the scalar MSE loss. (Provided.)"""
    y = x @ W.T + b
    return ((y - t) ** 2).mean()


def linear_mse_backward(
    x: Tensor, W: Tensor, b: Tensor, t: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Analytic gradients of linear_mse_forward w.r.t. W, b, x.
    Shapes: x (n, d_in), W (d_out, d_in), b (d_out,), t (n, d_out).

    TODO: With y = x @ W.T + b and the mean taken over n*d_out entries, return
    (dW, db, dx) where
        g  = (2 / (n * d_out)) * (y - t)     # dL/dy, shape (n, d_out)
        dW = g.T @ x                         # (d_out, d_in)
        db = g.sum(dim=0)                    # (d_out,)
        dx = g @ W                           # (n, d_in)
    """
    y = x @ W.T + b[None, :]
    err = y - t
    g = (2 / (err.shape[0] * err.shape[1])) * err  # (n, d_out)

    dW = g.T @ x  # (d_out, d_in)
    dx = g @ W  # (n, d_in)
    db = torch.einsum("ij -> j", g)  # (d_out,)
    return dW, db, dx


def finite_difference_grad(f, x: Tensor, eps: float = 1e-4) -> Tensor:
    """Central-difference gradient of scalar f at x. (Provided, for checking.)"""
    grad = torch.zeros_like(x)
    flat, g = x.reshape(-1), grad.reshape(-1)  # is this a view?
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        fp = f(x)
        flat[i] = orig - eps
        fm = f(x)
        flat[i] = orig
        g[i] = (fp - fm) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# Part 3: A training loop from scratch (no autograd)
# ---------------------------------------------------------------------------


def train_linear_regression(x: Tensor, t: Tensor, lr: float = 0.1, steps: int = 200):
    """
    Fit y = x @ W.T + b to targets t with plain SGD, using your analytic
    backward from Part 2 (NOT autograd). Return (W, b, loss_history).

    TODO:
      - Initialize W (d_out, d_in) and b (d_out,) to zeros.
      - Each step: loss = linear_mse_forward(...); (dW, db, _) = backward(...);
        update W and b in-place (W -= lr * dW, b -= lr * db); append loss.item().
    """
    W = torch.zeros((t.shape[1], x.shape[1]))
    b = torch.zeros((t.shape[1],))
    losses = []
    for _ in range(steps):
        loss = linear_mse_forward(x, W, b, t)
        dW, db, _ = linear_mse_backward(x, W, b, t)
        W -= lr * dW
        b -= lr * db
        losses.append(loss.item())
    return W, b, losses


if __name__ == "__main__":
    torch.manual_seed(0)

    # Part 1 checks
    X, Y = torch.randn(5, 3), torch.randn(4, 3)
    D_ref = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    print(
        "pairwise dist max err:",
        (pairwise_sq_distances(X, Y) - D_ref).abs().max().item(),
    )

    x, A, y = torch.randn(8, 3), torch.randn(8, 3, 4), torch.randn(8, 4)
    print(
        "bilinear max err:",
        (batched_bilinear(x, A, y) - _reference_bilinear(x, A, y)).abs().max().item(),
    )

    # Part 2: analytic vs autograd vs finite-difference (use double for accuracy)
    n, d_in, d_out = 6, 4, 3
    x = torch.randn(n, d_in, dtype=torch.double)
    W = torch.randn(d_out, d_in, dtype=torch.double, requires_grad=True)
    b = torch.randn(d_out, dtype=torch.double, requires_grad=True)
    t = torch.randn(n, d_out, dtype=torch.double)

    linear_mse_forward(x, W, b, t).backward()
    dW, db, dx = linear_mse_backward(x, W, b, t)
    print("dW err vs autograd:", (dW - W.grad).abs().max().item())
    print("db err vs autograd:", (db - b.grad).abs().max().item())
    fdx = finite_difference_grad(
        lambda xx: linear_mse_forward(xx, W.detach(), b.detach(), t), x.clone()
    )
    print("dx err vs finite-diff:", (dx - fdx).abs().max().item())

    # Part 3: training
    Wtrue = torch.randn(2, 4)
    xs = torch.randn(256, 4)
    ts = xs @ Wtrue.T + 0.05 * torch.randn(256, 2)
    W, b, hist = train_linear_regression(xs, ts)
    plt.plot(hist)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.title("From-scratch linear regression")
    plt.savefig("tensor_training.png", dpi=120)
    print(f"final loss: {hist[-1]:.4e}")
