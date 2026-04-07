# AI-generated
"""
dataset.py — Dataset stub for contrastive drone-vs-satellite training.

In a real deployment positive pairs would come from flights with GPS ground-truth:
    - Drone image  → warped to bird's eye via Task 1 pipeline
    - Satellite crop → extracted from the map at the known GPS position

Because no real data is available this stub generates *synthetic* pairs by:
    1. Sampling a random RGB patch (satellite anchor).
    2. Applying drone-style augmentation to a copy of the same patch (query).

This is enough to overfit the training loop on a single data point and verify
that gradients flow and the loss converges.

Replace `SyntheticPairDataset` with `RealPairDataset` (or extend this class)
once labelled drone+map data is available.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# ImageNet normalisation — matches the encoder's pretrained weights
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_to_tensor_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _drone_augmentation() -> Callable:
    """
    Augmentation pipeline applied to the bird's eye (drone) branch.

    Simulates the photometric and geometric variation introduced by:
        - Altitude changes ±20 m  → small random scale / crop
        - Pitch variation ±5°     → slight perspective-like shear
        - Exposure / lighting     → brightness, contrast, saturation jitter
    """
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.RandomAffine(degrees=0, shear=5),          # pitch proxy
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _satellite_augmentation() -> Callable:
    """
    Light augmentation applied to the satellite crop branch.

    Satellite imagery is relatively stable but we add:
        - Random rotation (map orientation may differ from drone heading)
        - Mild colour jitter (seasonal / sensor variation)
        - Small scale variation
    """
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.9, 1.0)),
        T.RandomRotation(degrees=180),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Synthetic dataset (stub)
# ---------------------------------------------------------------------------

class SyntheticPairDataset(Dataset):
    """
    Generates random (drone_view, satellite_crop) positive pairs on-the-fly.

    Both views are derived from the *same* randomly-sampled base image so the
    encoder must learn view-invariant representations to minimise InfoNCE loss.

    Args:
        num_samples: Number of synthetic pairs per epoch.
        image_size: Spatial size of the base random patch before augmentation.
                    Should be >= 256 so crops at 224 have some margin.
        seed: Optional RNG seed for reproducibility.

    Returns (per __getitem__):
        drone_view:  (3, 224, 224) tensor — augmented drone branch.
        sat_view:    (3, 224, 224) tensor — augmented satellite branch.
        pair_index:  int — index of the pair (useful for debugging).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 256,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.rng = random.Random(seed)

        self.drone_aug = _drone_augmentation()
        self.sat_aug = _satellite_augmentation()

        # Pre-generate base images as uint8 PIL-compatible tensors
        # (generated once; augmentation differs each time __getitem__ is called)
        torch.manual_seed(seed if seed is not None else 42)
        # Store as (N, H, W, 3) uint8 for PIL conversion
        self._bases: torch.Tensor = torch.randint(
            0, 256, (num_samples, image_size, image_size, 3), dtype=torch.uint8
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        from PIL import Image
        import numpy as np

        base_np = self._bases[idx].numpy()          # (H, W, 3) uint8
        base_pil = Image.fromarray(base_np, mode="RGB")

        drone_view = self.drone_aug(base_pil)        # (3, 224, 224)
        sat_view = self.sat_aug(base_pil)            # (3, 224, 224)

        return drone_view, sat_view, idx


# ---------------------------------------------------------------------------
# Real dataset scaffold (to be filled once data is available)
# ---------------------------------------------------------------------------

class RealPairDataset(Dataset):
    """
    Placeholder for a dataset backed by real drone footage + satellite tiles.

    Expected directory layout::

        data_root/
            pairs.csv          # columns: drone_path, sat_path, lat, lon
            drone/
                <flight_id>/   # bird's-eye warped images (800×800 PNG)
            satellite/
                <tile_id>.png  # 800×800 satellite crops at 1px=1m

    Args:
        data_root: Path to dataset root directory.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        drone_transform: Optional override for drone augmentation.
        sat_transform: Optional override for satellite augmentation.

    Raises:
        FileNotFoundError: If ``data_root`` or ``pairs.csv`` does not exist.
        NotImplementedError: Always — this class is a stub.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        drone_transform: Optional[Callable] = None,
        sat_transform: Optional[Callable] = None,
    ):
        raise NotImplementedError(
            "RealPairDataset is a stub. Provide real data and implement "
            "__getitem__ to load (drone_path, sat_path) from pairs.csv."
        )

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx: int):  # pragma: no cover
        raise NotImplementedError
