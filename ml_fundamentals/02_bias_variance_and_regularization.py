"""
Assignment 2: Bias–Variance & Regularization — Why Models Generalize
====================================================================
The bias–variance decomposition is quoted everywhere and implemented almost
nowhere. Here you measure it directly by Monte Carlo — training the same model
class on many resampled datasets — and watch ridge regularization trade bias
for variance. Along the way you hit two classic practical traps: regularizing
the bias term, and leaking test statistics through preprocessing during
cross-validation.

Topics:
  - Ridge regression in closed form (and why the intercept is NOT penalized)
  - Empirical bias^2 / variance / noise decomposition of expected test MSE
  - K-fold cross-validation from scratch
  - Data leakage: fitting the standardizer on all data vs per-fold

Refs:
  - Hastie, Tibshirani, Friedman — Elements of Statistical Learning, ch. 7
  - Belkin et al., Reconciling modern ML and the bias-variance trade-off
    https://arxiv.org/abs/1812.11118

Setup: pip install torch numpy matplotlib   (all in the lean env)
"""

import matplotlib.pyplot as plt
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Ground truth: a smooth 1-D function with observation noise. Because we KNOW
# the generator, we can estimate bias and variance exactly.
# ---------------------------------------------------------------------------

NOISE_STD = 0.3


def true_fn(x: Tensor) -> Tensor:
    return torch.sin(2.5 * x) + 0.5 * x


def sample_dataset(n: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Draw x ~ U[-2, 2], y = f(x) + eps, eps ~ N(0, NOISE_STD^2). (Provided.)"""
    x = 4 * torch.rand(n, generator=generator) - 2
    y = true_fn(x) + NOISE_STD * torch.randn(n, generator=generator)
    return x, y


# ---------------------------------------------------------------------------
# Part 1: Polynomial ridge regression in closed form
# ---------------------------------------------------------------------------


def poly_features(x: Tensor, degree: int) -> Tensor:
    """
    Map x (n,) to the design matrix Phi (n, degree + 1) with columns
    [1, x, x^2, ..., x^degree].

    TODO: One expression with broadcasting — x[:, None] raised to a
    torch.arange of exponents. No loop.
    """
    exponents = torch.arange(degree+1)
    return x[:, None] ** exponents


def ridge_fit(Phi: Tensor, y: Tensor, lam: float) -> Tensor:
    """
    Closed-form ridge: w = (Phi^T Phi + lam * I')^{-1} Phi^T y, shape (p,).

    TODO:
      - I' is the identity EXCEPT I'[0, 0] = 0: the intercept column (all
        ones, column 0 of poly_features) must not be shrunk. Penalizing it
        makes predictions depend on the arbitrary origin of y — verify later
        in __main__ that shifting all y by +100 changes nothing but w[0].
      - Use torch.linalg.solve, not an explicit inverse (better conditioned;
        high-degree polynomial Gram matrices are nearly singular).
      - Work in float64: Phi^T Phi for degree ~10 already loses most float32
        precision.
    """
    # Phi n, m
    # I - m, m
    # y = n, out_d
    I = torch.eye(Phi.shape[-1], dtype=torch.float64)
    I[0,0] = 0.
    Phi = Phi.to(torch.float64)
    uninverted = Phi.T @ Phi + lam * I  # m, m
    res = torch.linalg.solve(uninverted, Phi.T @ y)
    return res.to(torch.float32)


def ridge_predict(Phi: Tensor, w: Tensor) -> Tensor:
    """TODO: one line."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: Measuring bias^2 and variance by Monte Carlo
# ---------------------------------------------------------------------------
# For a fixed test point x*, over the randomness of the TRAINING SET D:
#
#   E_D[(y* - f_D(x*))^2] = (f(x*) - E_D[f_D(x*)])^2   <- bias^2
#                         + E_D[(f_D(x*) - E_D[f_D(x*)])^2]   <- variance
#                         + NOISE_STD^2                 <- irreducible noise
#
# We estimate the two E_D terms by training on `n_trials` freshly sampled
# datasets and collecting each model's predictions on a shared test grid.


def bias_variance_estimate(
    degree: int,
    lam: float,
    n_train: int = 30,
    n_trials: int = 300,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Return (avg_bias_sq, avg_variance), each averaged over a fixed test grid
    x_grid = torch.linspace(-2, 2, 100).

    TODO:
      - Loop n_trials times: sample a fresh training set (one shared
        torch.Generator, do NOT reseed per trial — reseeding gives you the
        same dataset 300 times and variance == 0, a bug that looks like a
        result), fit ridge on poly features, predict on the grid. Stack
        predictions into P (n_trials, 100).
      - mean_pred = P.mean(0)
        bias_sq   = ((true_fn(x_grid) - mean_pred) ** 2).mean()
        variance  = P.var(0, unbiased=True).mean()
      - Sanity: for degree=1 bias should dominate; for degree=12, lam=0
        variance should dominate.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: K-fold cross-validation — and the leakage trap
# ---------------------------------------------------------------------------


def kfold_indices(n: int, k: int, generator: torch.Generator) -> list[tuple[Tensor, Tensor]]:
    """
    Return k (train_idx, val_idx) pairs of long tensors covering a random
    permutation of range(n), folds as equal as possible.

    TODO: torch.randperm(n, generator=generator), then torch.chunk into k
    validation folds; each train_idx is the complement (torch.cat of the
    other chunks).
    """
    raise NotImplementedError


def cv_mse(
    x: Tensor, y: Tensor, degree: int, lam: float, k: int, generator: torch.Generator,
    leaky: bool = False,
) -> float:
    """
    Mean validation MSE over k folds for polynomial ridge, where inputs are
    standardized feature-wise (columns of Phi except the intercept get
    (col - mean) / std).

    TODO — this is the point of the exercise:
      - leaky=False (correct): compute the standardization mean/std on the
        TRAINING FOLD ONLY, apply to both train and val features.
      - leaky=True (the bug): compute mean/std on ALL of Phi before
        splitting. The validation score now uses statistics that saw the
        validation targets' inputs — a mild but real form of leakage that
        systematically flatters the score.
      - Guard std with a 1e-12 floor. Fit with ridge_fit, evaluate MSE on
        the val fold, average over folds.
    """
    raise NotImplementedError


def select_lambda(
    x: Tensor, y: Tensor, degree: int, lambdas: list[float], k: int = 5, seed: int = 0
) -> tuple[float, list[float]]:
    """
    Pick the lambda with the lowest (non-leaky) CV MSE.
    Return (best_lambda, all_scores).

    TODO: build ONE set of folds (one generator) and reuse it for every
    lambda — re-randomizing folds per candidate adds noise exactly where you
    are trying to compare small differences.
    """
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    # ---- Part 1 checks ------------------------------------------------------
    g = torch.Generator().manual_seed(1)
    x, y = sample_dataset(60, g)
    Phi = poly_features(x, degree=3)
    assert Phi.shape == (60, 4) and torch.allclose(Phi[:, 0], torch.ones(60))

    w = ridge_fit(Phi, y, lam=1.0)
    w_shift = ridge_fit(Phi, y + 100.0, lam=1.0)
    assert torch.allclose(w[1:], w_shift[1:], atol=1e-6), (
        "shifting y changed non-intercept weights -> you penalized the intercept"
    )
    print("intercept-penalty check: OK")

    # lam -> huge shrinks slope weights to ~0 but NOT the intercept
    w_big = ridge_fit(Phi, y, lam=1e9)
    assert w_big[1:].abs().max() < 1e-3 and w_big[0].abs() > 1e-2
    print("shrinkage check: OK")

    # ---- Part 2: bias/variance vs degree ------------------------------------
    degrees = list(range(1, 13))
    stats = [bias_variance_estimate(d, lam=0.0) for d in degrees]
    bias2 = [s[0] for s in stats]
    var = [s[1] for s in stats]
    total = [b + v + NOISE_STD**2 for b, v in zip(bias2, var)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(degrees, bias2, marker="o", label="bias$^2$")
    axes[0].plot(degrees, var, marker="s", label="variance")
    axes[0].plot(degrees, total, marker="^", label="total (+noise)")
    axes[0].axhline(NOISE_STD**2, ls="--", c="gray", lw=1, label="noise floor")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("polynomial degree")
    axes[0].set_title("Bias-variance decomposition (lam=0)")
    axes[0].legend()

    # Regularizing a high-degree model: variance falls, bias rises.
    lams = [0.0, 1e-4, 1e-2, 1.0, 100.0]
    stats_l = [bias_variance_estimate(10, lam=l) for l in lams]
    axes[1].plot(range(len(lams)), [s[0] for s in stats_l], marker="o", label="bias$^2$")
    axes[1].plot(range(len(lams)), [s[1] for s in stats_l], marker="s", label="variance")
    axes[1].set_xticks(range(len(lams)), [str(l) for l in lams])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("lambda (degree 10)")
    axes[1].set_title("Ridge trades variance for bias")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("bias_variance.png", dpi=120)
    print("saved bias_variance.png")

    # ---- Part 3: CV and leakage ---------------------------------------------
    best_lam, scores = select_lambda(x, y, degree=10, lambdas=lams)
    print(f"CV-selected lambda: {best_lam}")

    g2 = torch.Generator().manual_seed(3)
    honest = cv_mse(x, y, 10, best_lam, k=5, generator=g2, leaky=False)
    g2 = torch.Generator().manual_seed(3)
    leaked = cv_mse(x, y, 10, best_lam, k=5, generator=g2, leaky=True)
    print(f"CV MSE honest: {honest:.4f}   leaky: {leaked:.4f}")
    print("(leaky is typically lower — an optimistic lie, not a better model)")
