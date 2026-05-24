"""
Assignment 1: Contrastive Learning — InfoNCE & CLIP from Scratch
================================================================
Contrastive learning is the backbone of CLIP, DINO, and most modern
vision-language models. For an RL researcher it's also directly relevant:
contrastive loss appears in representation learning for RL (CURL, SPR,
ATC) and provides a principled way to learn state embeddings.

Topics:
  - InfoNCE / NT-Xent loss and its connection to mutual information
  - In-batch negatives and why large batch size matters
  - Temperature τ as a sharpness parameter (relate to entropy in RL)
  - Symmetric vs. asymmetric objectives
  - Linear probing to evaluate learned representations

Paper refs:
  - CPC / InfoNCE (Oord et al. 2018)  https://arxiv.org/abs/1807.03748
  - SimCLR (Chen et al. 2020)  https://arxiv.org/abs/2002.05709
  - CLIP (Radford et al. 2021)  https://arxiv.org/abs/2103.00020
  - CURL (Laskin et al. 2020)  https://arxiv.org/abs/2004.04136

Setup: pip install torch torchvision matplotlib scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

# ---------------------------------------------------------------------------
# Part 1: InfoNCE loss — the core of contrastive learning
# ---------------------------------------------------------------------------

def infonce_loss(z_i: Tensor, z_j: Tensor, tau: float = 0.07) -> Tensor:
    """
    InfoNCE (NT-Xent) with in-batch negatives.
    z_i, z_j: L2-normalized embeddings of shape (B, d), paired augmentations.

    Loss for anchor z_i[k]:
      L_k = -log [ exp(z_i[k]·z_j[k]/τ) / Σ_{m≠k} exp(z_i[k]·z_j[m]/τ) ]

    The symmetric version averages L(i→j) and L(j→i).

    TODO: Implement using the (2B × 2B) similarity matrix trick.
    Mask out self-similarities on the diagonal.

    Hint: torch.cat([z_i, z_j], dim=0) gives you the full 2B set.
    The positives are at off-diagonal positions (k, k+B) and (k+B, k).
    """
    raise NotImplementedError


def temperature_analysis(z_i: Tensor, z_j: Tensor) -> None:
    """
    TODO: Plot InfoNCE loss vs. τ ∈ [0.01, 0.05, 0.1, 0.2, 0.5, 1.0].
    Relate: low τ → peaked distribution → high-entropy targets (RL entropy bonus
    analogy), high τ → uniform → insufficient signal.
    Also plot gradient norm vs. τ to show why τ too small causes instability.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: CLIP-style dual-encoder architecture
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """ResNet-18 backbone → projection head → L2-normalized embedding."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # TODO: Add MLP projection head: 512 → 512 → embed_dim, with BN+ReLU
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Return L2-normalized embedding."""
        raise NotImplementedError


class TextEncoder(nn.Module):
    """
    Toy text encoder: bag-of-words embedding → MLP → L2-normalized embedding.
    (Real CLIP uses a Transformer; this is for pedagogical speed.)
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        # TODO: nn.EmbeddingBag (mean pooling) → MLP → embed_dim
        raise NotImplementedError

    def forward(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError


class CLIP(nn.Module):
    """Dual-encoder with learnable temperature logit_scale = log(1/τ)."""

    def __init__(self, image_encoder: ImageEncoder, text_encoder: TextEncoder):
        super().__init__()
        self.image_enc = image_encoder
        self.text_enc = text_encoder
        # CLIP learns log(1/τ) as a scalar parameter, clipped to [0, log(100)]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        TODO: Return (image_embeddings, text_embeddings), both L2-normalized.
        """
        raise NotImplementedError

    def loss(self, images: Tensor, tokens: Tensor) -> Tensor:
        """
        Symmetric cross-entropy over cosine similarities scaled by exp(logit_scale).

        TODO: Compute NxN logit matrix, then CE loss in both directions.
        Ground truth is the diagonal (image i matches text i).
        Clip logit_scale to [0, log(100)] before exponentiation.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: SimCLR training on CIFAR-10 (image-only contrastive)
# ---------------------------------------------------------------------------

def get_augmentation_pair() -> transforms.Compose:
    """
    SimCLR augmentation: RandomResizedCrop + ColorJitter + GrayScale + GaussianBlur.
    Return as a pair-producing transform.

    TODO: Implement. Strength of augmentation critically affects representation quality.
    """
    raise NotImplementedError


class PairedDataset(torch.utils.data.Dataset):
    """Wraps CIFAR-10 and returns two augmented views of the same image."""

    def __init__(self, root: str, train: bool, aug: transforms.Compose):
        self.base = datasets.CIFAR10(root, train=train, download=True,
                                     transform=transforms.ToTensor())
        self.aug = aug

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        img, label = self.base[idx]
        return self.aug(img), self.aug(img), label

    def __len__(self) -> int:
        return len(self.base)


def train_simclr(epochs: int = 10, batch_size: int = 256, tau: float = 0.07):
    """
    TODO: Train ImageEncoder on CIFAR-10 using InfoNCE.
    Log loss per epoch. After training, evaluate with linear_probe().
    Return the trained encoder.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: Linear probing — standard representation evaluation
# ---------------------------------------------------------------------------

def linear_probe(encoder: nn.Module, data_root: str = "./data") -> float:
    """
    Freeze encoder, extract features from train set, fit a linear classifier,
    report test accuracy. This is the standard eval for self-supervised methods.

    TODO:
      1. Extract embeddings for all train/test images (no augmentation).
      2. Fit sklearn LogisticRegression on train embeddings.
      3. Return test accuracy.

    Compare against a randomly initialized (untrained) encoder as baseline.
    """
    raise NotImplementedError


if __name__ == "__main__":
    encoder = train_simclr(epochs=10)
    acc = linear_probe(encoder)
    print(f"Linear probe accuracy: {acc:.3f}")
    print("(Random baseline ~10%. Well-trained SimCLR on CIFAR-10 reaches ~85%.)")
