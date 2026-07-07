"""
Assignment 3: Logistic Regression & the Numerics Everyone Gets Wrong
====================================================================
Logistic regression is the simplest model whose naive implementation is
numerically broken: sigmoid overflows, log(sigmoid) underflows, and softmax
explodes without the max-subtraction trick. This assignment makes you build
the stable versions from scratch, derive the beautifully simple gradient
(p - y), and see why separable data drives unregularized weights to infinity.

Topics:
  - Numerically stable log-sigmoid and binary cross-entropy FROM LOGITS
  - Softmax cross-entropy via log-sum-exp; gradient = (softmax(z) - onehot)/n
  - Full-batch gradient descent for multiclass logistic regression
  - Weight divergence on linearly separable data, fixed by L2
  - Verifying hand-derived gradients against torch.autograd

Refs:
  - Murphy, Probabilistic Machine Learning ch. 10 (logistic regression)
  - What every computer scientist should know about floating-point
    https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

Setup: pip install torch numpy matplotlib scikit-learn   (all in the lean env)
"""

import matplotlib.pyplot as plt
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Part 1: Stable binary cross-entropy from logits
# ---------------------------------------------------------------------------
# The naive route  p = 1/(1+exp(-z)); loss = -[y log p + (1-y) log(1-p)]
# fails twice: exp(-z) overflows for z < -100ish, and log(p) returns -inf
# once p rounds to 0 (which happens for z < -40 in float32). The fix is to
# never form p at all.


def stable_log_sigmoid(z: Tensor) -> Tensor:
    """
    log(sigmoid(z)), elementwise, stable for all z in float32.

    TODO: Use the identity  log sigmoid(z) = -log(1 + exp(-z)) = -softplus(-z)
    and the stable softplus  softplus(a) = max(a, 0) + log1p(exp(-|a|)).
    Implement softplus yourself with torch.clamp / torch.log1p — do NOT call
    torch.nn.functional.{softplus,logsigmoid} (that's the answer key).
    Check: z = -1000 must give -1000.0 (not -inf), z = 1000 must give 0.0.
    """
    raise NotImplementedError


def bce_from_logits(z: Tensor, y: Tensor) -> Tensor:
    """
    Mean binary cross-entropy from logits z (n,) and labels y (n,) in {0, 1}.

    TODO: loss_i = -[ y_i * log_sigmoid(z_i) + (1 - y_i) * log_sigmoid(-z_i) ]
    using stable_log_sigmoid (note log(1 - sigmoid(z)) = log_sigmoid(-z)).
    Return the mean. __main__ checks you match
    torch.nn.functional.binary_cross_entropy_with_logits at z = ±500, where
    the naive version returns nan/inf.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Softmax cross-entropy and its gradient
# ---------------------------------------------------------------------------


def log_softmax(Z: Tensor) -> Tensor:
    """
    Row-wise log-softmax of logits Z (n, k).

    TODO: log_softmax(z) = z - logsumexp(z). Implement logsumexp yourself
    with the max trick:
        m = Z.max(dim=1, keepdim=True).values
        logsumexp = m + log(sum(exp(Z - m)))
    (torch.logsumexp is the answer key — don't call it here.)
    Subtracting the max is exact, not approximate: exp(z - m) <= 1 never
    overflows, and at least one entry equals 1 so the sum never underflows.
    """
    raise NotImplementedError


def softmax_xent(Z: Tensor, y: Tensor) -> Tensor:
    """
    Mean cross-entropy for logits Z (n, k) and integer labels y (n,).

    TODO: mean over i of -log_softmax(Z)[i, y[i]]. Pick out the label column
    with torch.gather or fancy indexing — no one_hot matmul needed.
    """
    raise NotImplementedError


def softmax_xent_grad(Z: Tensor, y: Tensor) -> Tensor:
    """
    Analytic gradient dL/dZ of softmax_xent, shape (n, k).

    TODO: the celebrated result — softmax and cross-entropy's Jacobians
    cancel almost entirely:
        dL/dZ = (softmax(Z) - onehot(y)) / n
    Derive it once on paper before typing it. __main__ checks it against
    autograd to ~1e-6.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: Multiclass logistic regression by gradient descent
# ---------------------------------------------------------------------------
# Model: Z = X @ W + b, W (d, k), b (k,). By the chain rule, with
# G = softmax_xent_grad(Z, y):
#     dW = X.T @ G + lam * W        (L2 penalty on W only, never on b)
#     db = G.sum(0)


def train_logreg(
    X: Tensor,
    y: Tensor,
    k: int,
    lr: float = 0.5,
    steps: int = 500,
    lam: float = 0.0,
) -> tuple[Tensor, Tensor, list[float], list[float]]:
    """
    Full-batch gradient descent. Return (W, b, loss_history, wnorm_history)
    where wnorm_history tracks ||W|| each step.

    TODO:
      - Init W, b to zeros ((d, k) and (k,)). Zeros are FINE here (unlike in
        an MLP): the model is linear in its parameters, there is no hidden
        symmetry to break.
      - Each step: Z = X @ W + b; record softmax_xent (report the loss
        WITHOUT the penalty term so runs with different lam are comparable);
        compute (dW, db) as above; SGD update; record W.norm().item().
      - No autograd anywhere in this function — that's the point.
    """
    raise NotImplementedError


def predict(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    """TODO: argmax over classes, shape (n,)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Separable data and why regularization is not optional
# ---------------------------------------------------------------------------
# On linearly separable data the unregularized MLE does not exist: scaling
# any separating W by c > 1 strictly lowers the loss, so ||W|| -> inf and
# the probabilities saturate to 0/1. Any lam > 0 restores a finite optimum.
# You demonstrate this empirically in __main__ using wnorm_history — no new
# code to write here, but predict what the two curves will look like BEFORE
# you run it.


if __name__ == "__main__":
    import torch.nn.functional as F
    from sklearn.datasets import make_moons

    torch.manual_seed(0)

    # ---- Part 1: stability checks -------------------------------------------
    z_extreme = torch.tensor([-500.0, -30.0, 0.0, 30.0, 500.0])
    ls = stable_log_sigmoid(z_extreme)
    assert torch.isfinite(ls).all(), "log-sigmoid produced inf/nan"
    assert torch.allclose(ls, F.logsigmoid(z_extreme), atol=1e-6)
    print("stable log-sigmoid: OK")

    y_bin = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    ref = F.binary_cross_entropy_with_logits(z_extreme, y_bin)
    assert torch.allclose(bce_from_logits(z_extreme, y_bin), ref, atol=1e-5)
    naive = -(y_bin * torch.log(torch.sigmoid(z_extreme))
              + (1 - y_bin) * torch.log(1 - torch.sigmoid(z_extreme))).mean()
    print(f"stable BCE: {bce_from_logits(z_extreme, y_bin):.2f}   naive: {naive}")

    # ---- Part 2: gradient check ---------------------------------------------
    Z = (100 * torch.randn(16, 5)).requires_grad_()  # big logits on purpose
    y_mc = torch.randint(0, 5, (16,))
    assert torch.allclose(log_softmax(Z), F.log_softmax(Z, dim=1), atol=1e-5)
    softmax_xent(Z, y_mc).backward()
    G = softmax_xent_grad(Z.detach(), y_mc)
    print("xent grad err vs autograd:", (G - Z.grad).abs().max().item())

    # ---- Part 3: train on two moons -----------------------------------------
    Xm_np, ym_np = make_moons(n_samples=400, noise=0.2, random_state=0)
    Xm = torch.from_numpy(Xm_np).float()
    ym = torch.from_numpy(ym_np).long()
    # Quadratic features make the moons (nearly) separable by a linear model.
    Xf = torch.cat([Xm, Xm**2, (Xm[:, 0] * Xm[:, 1])[:, None]], dim=1)

    W, b, hist, wn = train_logreg(Xf, ym, k=2, lam=1e-3)
    acc = (predict(Xf, W, b) == ym).float().mean()
    print(f"train accuracy: {acc:.3f}  (expect > 0.95)")

    # ---- Part 4: weight blow-up without regularization ----------------------
    _, _, _, wn0 = train_logreg(Xf, ym, k=2, lam=0.0, steps=3000)
    _, _, _, wnr = train_logreg(Xf, ym, k=2, lam=1e-3, steps=3000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(wn0, label="lam = 0 (diverges)")
    axes[0].plot(wnr, label="lam = 1e-3 (converges)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("||W||")
    axes[0].set_title("Separable data: MLE runs away")
    axes[0].legend()

    # Decision boundary of the regularized model
    xx, yy = torch.meshgrid(
        torch.linspace(-1.8, 2.8, 200), torch.linspace(-1.3, 1.8, 200), indexing="xy"
    )
    Gpts = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    Gf = torch.cat([Gpts, Gpts**2, (Gpts[:, 0] * Gpts[:, 1])[:, None]], dim=1)
    zz = predict(Gf, W, b).reshape(xx.shape)
    axes[1].contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    axes[1].scatter(Xm[:, 0], Xm[:, 1], c=ym, s=10, cmap="coolwarm")
    axes[1].set_title("Decision boundary (quadratic features)")
    fig.tight_layout()
    fig.savefig("logreg_numerics.png", dpi=120)
    print("saved logreg_numerics.png")
