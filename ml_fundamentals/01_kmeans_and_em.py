"""
Assignment 1: K-Means & EM — Clustering from First Principles
=============================================================
K-means looks trivial until you implement it: vectorizing the assignment step,
seeding it well (k-means++), handling empty clusters, and recognizing it as a
degenerate case of EM for a Gaussian mixture are all classic interview and
exam territory. This assignment builds Lloyd's algorithm from scratch, then
generalizes it to soft assignments with EM for a spherical GMM.

Topics:
  - Lloyd's algorithm: alternating assignment / update as coordinate descent
    on the distortion J = sum_i ||x_i - mu_{z_i}||^2
  - k-means++ initialization and why random init gets stuck
  - Empty-cluster handling (a detail every from-scratch version hits)
  - Model selection: inertia vs k, the elbow heuristic
  - EM for a spherical GMM: responsibilities via log-sum-exp, and the limit
    sigma -> 0 recovering hard k-means

Refs:
  - k-means++: The Advantages of Careful Seeding (Arthur & Vassilvitskii)
    https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
  - Bishop, PRML ch. 9 (Mixture Models and EM)
  - Neal & Hinton, A View of the EM Algorithm  https://www.cs.toronto.edu/~radford/ftp/emk.pdf

Setup: pip install torch numpy matplotlib scikit-learn   (all in the lean env)
"""

import matplotlib.pyplot as plt
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Part 1: The assignment step — vectorized nearest-centroid search
# ---------------------------------------------------------------------------


def pairwise_sq_distances(X: Tensor, C: Tensor) -> Tensor:
    """
    Squared Euclidean distances between points X (n, d) and centroids C (k, d).
    Return D of shape (n, k) with D[i, j] = ||X[i] - C[j]||^2.

    TODO: No Python loops and no (n, k, d) intermediate. Use
        ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
    (You solved the same problem in pytorch_foundations/01 — redo it from
    memory.) Clamp the result at 0: floating-point cancellation can make
    tiny distances slightly negative.
    """
    x2 = torch.einsum("ik, ik -> i", X, X)
    xc = torch.einsum("ik, jk -> ij", X, C)
    c2 = torch.einsum("jk, jk -> j", C, C)
    return torch.clamp(x2[:, None] - 2 * xc + c2[None, :], min=0.0)


def assign_clusters(X: Tensor, C: Tensor) -> Tensor:
    """
    Return z of shape (n,), dtype long: z[i] = argmin_j ||X[i] - C[j]||^2.

    TODO: One line on top of pairwise_sq_distances.
    """
    return torch.argmin(pairwise_sq_distances(X, C), dim=-1)


# ---------------------------------------------------------------------------
# Part 2: The update step — and the empty-cluster gotcha
# ---------------------------------------------------------------------------


def update_centroids(X: Tensor, z: Tensor, k: int) -> Tensor:
    """
    Given points X (n, d) and hard assignments z (n,), return new centroids
    (k, d) where row j is the mean of the points assigned to cluster j.

    TODO:
      - Vectorize the per-cluster mean. One clean way: build a one-hot matrix
        H (n, k) with torch.nn.functional.one_hot, then
            counts = H.sum(0)            # (k,)
            sums   = H.T.float() @ X     # (k, d)
            C      = sums / counts[:, None]
      - Empty clusters: if counts[j] == 0 the mean is NaN. Re-seed any empty
        centroid to the data point currently FARTHEST from its own assigned
        centroid (a standard fix that also lowers J). Don't skip this — the
        __main__ check triggers it deliberately.
    """
    one_hot = torch.nn.functional.one_hot(z, k)  # N, k
    counts = one_hot.sum(0)  # k
    sums = one_hot.T @ X  # k, d
    C = sums / counts[:, None]
    empty = (counts == 0).nonzero(as_tuple=True)[0]
    if empty.sum() > 0:
        self_dist = ((X - C[z]) ** 2).sum(-1)
        new_idx = torch.topk(self_dist, empty.shape[0]).indices
        C[empty] = X[new_idx]
    return C


def inertia(X: Tensor, C: Tensor, z: Tensor) -> float:
    """
    Distortion J = sum_i ||X[i] - C[z[i]]||^2 (a plain float).

    TODO: index the distance matrix (or compute residuals directly) — no loop.
    """
    return ((X - C[z]) ** 2).sum().item()


# ---------------------------------------------------------------------------
# Part 3: k-means++ initialization
# ---------------------------------------------------------------------------
# Random init frequently drops two centroids into one true cluster and none
# into another; Lloyd's can't recover because it only moves centroids locally.
# k-means++ seeds far-apart centroids and carries an O(log k) approximation
# guarantee in expectation.


def kmeans_pp_init(X: Tensor, k: int, generator: torch.Generator) -> Tensor:
    """
    Return initial centroids (k, d) chosen by k-means++:
      1. Pick the first centroid uniformly at random from X.
      2. For each subsequent centroid, pick x_i with probability proportional
         to D(x_i)^2, where D(x_i) is the distance from x_i to the NEAREST
         centroid chosen so far.

    TODO:
      - Maintain d2 (n,) = squared distance to nearest chosen centroid;
        after adding a centroid, update with torch.minimum — don't recompute
        against all centroids each round.
      - Sample indices with torch.multinomial(d2, 1, generator=generator).
      - Use `generator` for ALL randomness so runs are reproducible.
    """
    cs = []
    assert k > 0, "no valid centroids to return"
    cs.append(
        X[
            torch.multinomial(
                torch.ones(X.shape[0], dtype=torch.long), 1, generator=generator
            )
        ]
    )
    d2 = ((X - cs[-1][None, :]) ** 2).sum(-1)
    for i in range(1, k):
        cs.append(X[torch.multinomial(d2, 1, generator=generator)])
        d2 = torch.minimum(d2, ((X - cs[-1][None, :]) ** 2).sum(-1))
    return torch.tensor(cs)


# ---------------------------------------------------------------------------
# Part 4: Lloyd's algorithm — putting it together
# ---------------------------------------------------------------------------


def kmeans(
    X: Tensor,
    k: int,
    n_init: int = 5,
    max_iters: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[Tensor, Tensor, float]:
    """
    Full k-means with k-means++ seeding and restarts.
    Return (C, z, J): centroids (k, d), assignments (n,), final inertia.

    TODO:
      - Run `n_init` independent restarts (fresh k-means++ seeding each time,
        offset the generator seed per restart) and keep the run with the
        lowest final inertia. This is how sklearn fights local minima.
      - Each run alternates assign_clusters / update_centroids until the
        relative inertia improvement (J_prev - J) / max(J_prev, 1e-12) < tol
        or max_iters is hit.
      - Key property to convince yourself of: BOTH steps can only decrease J
        (assignment minimizes J over z with C fixed; the mean minimizes J
        over C with z fixed), so J is monotone non-increasing — assert it
        while iterating, it will catch most bugs.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5: From hard to soft — EM for a spherical GMM
# ---------------------------------------------------------------------------
# Model: p(x) = sum_j pi_j N(x | mu_j, sigma_j^2 I). K-means is the limit of
# EM when all sigma_j -> 0 and pi_j are equal: responsibilities collapse to
# one-hot argmax. Implementing EM makes that connection concrete and forces
# you through the log-sum-exp trick.


def gmm_e_step(
    X: Tensor, pi: Tensor, mu: Tensor, sigma2: Tensor
) -> tuple[Tensor, float]:
    """
    E-step for a spherical GMM. Shapes: X (n, d), pi (k,), mu (k, d),
    sigma2 (k,) — per-component isotropic variances.
    Return (R, ll): responsibilities R (n, k) with rows summing to 1, and the
    total log-likelihood ll = sum_i log p(x_i) as a float.

    TODO: Work in log space throughout.
        log N(x | mu_j, sigma2_j I) = -d/2 * log(2*pi*sigma2_j)
                                      - ||x - mu_j||^2 / (2*sigma2_j)
        log_w[i, j] = log pi_j + log N(x_i | ...)
        ll_i        = logsumexp_j(log_w[i, :])      # torch.logsumexp
        R           = exp(log_w - ll_i[:, None])
    Never exponentiate log_w directly — with small sigma2 it underflows to
    all zeros and R becomes NaN. That failure mode is exactly what log-sum-exp
    exists to prevent.
    """
    raise NotImplementedError


def gmm_m_step(X: Tensor, R: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    M-step: given responsibilities R (n, k), return updated (pi, mu, sigma2).

    TODO: With N_j = R[:, j].sum():
        pi_j     = N_j / n
        mu_j     = (R[:, j][:, None] * X).sum(0) / N_j       # or R.T @ X
        sigma2_j = sum_i R[i, j] * ||x_i - mu_j||^2 / (N_j * d)
    Floor sigma2 at 1e-6 so a component collapsing onto a single point
    doesn't drive the likelihood to infinity (the classic GMM singularity).
    """
    raise NotImplementedError


def gmm_fit(
    X: Tensor, k: int, iters: int = 50, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, list[float]]:
    """
    Run EM for `iters` iterations. Return (pi, mu, sigma2, ll_history).

    TODO:
      - Initialize mu with kmeans_pp_init, pi uniform, sigma2 to the overall
        data variance (X.var()) for every component.
      - Alternate gmm_e_step / gmm_m_step, appending ll each iteration.
      - EM guarantees ll is monotone non-decreasing — assert it (with a small
        1e-6 slack for float error). If the assert fires, your E or M step
        is wrong.
    """
    raise NotImplementedError


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    torch.manual_seed(0)

    # ---- Parts 1-2: mechanics checks -------------------------------------
    X = torch.randn(50, 3)
    C = torch.randn(4, 3)
    D_ref = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    print(
        "pairwise dist max err:",
        (pairwise_sq_distances(X, C) - D_ref).abs().max().item(),
    )

    z = assign_clusters(X, C)
    assert z.shape == (50,) and z.dtype == torch.long

    # Empty-cluster trigger: 3 clusters requested, assignments only use {0, 1}
    z_forced = torch.tensor([0] * 25 + [1] * 25)
    C_new = update_centroids(X, z_forced, k=3)
    assert not torch.isnan(C_new).any(), "empty cluster produced NaN centroid"
    print("empty-cluster handling: OK")

    # ---- Parts 3-4: full k-means on blobs ---------------------------------
    Xb_np, _ = make_blobs(n_samples=600, centers=5, cluster_std=1.2, random_state=7)
    Xb = torch.from_numpy(Xb_np).float()

    C5, z5, J5 = kmeans(Xb, k=5)
    print(f"k=5 inertia: {J5:.1f}")

    # Compare against sklearn — you should land within ~1% of its inertia.
    from sklearn.cluster import KMeans

    sk = KMeans(n_clusters=5, n_init=5, random_state=0).fit(Xb_np)
    print(f"sklearn inertia: {sk.inertia_:.1f}  (ratio {J5 / sk.inertia_:.3f})")

    # Elbow plot: inertia always decreases with k; the "elbow" near the true
    # k=5 is the heuristic signal.
    ks = list(range(1, 10))
    Js = [kmeans(Xb, k=kk)[2] for kk in ks]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(Xb[:, 0], Xb[:, 1], c=z5, s=8, cmap="tab10")
    axes[0].scatter(C5[:, 0], C5[:, 1], marker="x", s=120, c="black")
    axes[0].set_title("k-means (k=5)")
    axes[1].plot(ks, Js, marker="o")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("inertia")
    axes[1].set_title("Elbow plot")

    # ---- Part 5: EM vs k-means --------------------------------------------
    pi, mu, sigma2, ll_hist = gmm_fit(Xb, k=5)
    axes[2].plot(ll_hist)
    axes[2].set_xlabel("EM iteration")
    axes[2].set_ylabel("log-likelihood")
    axes[2].set_title("EM monotone ascent")
    fig.tight_layout()
    fig.savefig("kmeans_em.png", dpi=120)
    print("GMM means recovered (compare to k-means centroids):")
    print(torch.sort(mu[:, 0])[0].round(decimals=1))
    print(torch.sort(C5[:, 0])[0].round(decimals=1))
    print("saved kmeans_em.png")
