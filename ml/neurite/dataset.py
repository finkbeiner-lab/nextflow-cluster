#!/usr/bin/env python3
"""Dataset for rasterized neurite ``(image, mask)`` pairs.

Loads the ``.npy`` pairs produced by ``rasterize_traces.py`` and serves
augmented crops for training. Two concerns dominate here and both stem from the
data being tiny (currently 2 annotated 2048x2048 fields):

1. **Percentile normalization (1-99.5).** The morphology channel is faint,
   16-bit, and background-dominated. We map the [1, 99.5] percentile window to
   [0, 1], the SAME normalization the Frangi tracer and F1 scorer use
   (``score_all.py``), so the network sees the intensities the benchmark was
   built on. Computed once per field at load time.

2. **Heavy augmentation + random crops.** With so few fields, each ``__getitem__``
   samples a random crop and applies random flips and 90-degree rotations (which
   are label-preserving for a binary mask). ``length`` decouples epoch size from
   the number of fields so an "epoch" is a fixed number of sampled crops. A
   deterministic ``val`` mode returns fixed centre crops for stable evaluation.

Splitting is done by CALLER at the field level (see ``train.py``) to avoid
train/val leakage between crops of the same field.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def percentile_normalize(
    image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5
) -> np.ndarray:
    """Normalize an image to [0, 1] using a percentile window.

    Matches the tracer/scorer normalization (``score_all.py``): clip to
    non-negative, then map [lo_pct, hi_pct] percentiles to [0, 1].

    Args:
        image: Input image, any real dtype.
        lo_pct: Lower percentile mapped to 0.
        hi_pct: Upper percentile mapped to 1.

    Returns:
        Float32 image clipped to [0, 1].
    """
    x = np.clip(image.astype(np.float32), 0, None)
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi > lo:
        return np.clip((x - lo) / (hi - lo), 0, 1)
    return np.zeros_like(x)


def list_pairs(data_dir: str) -> List[Tuple[str, str, str]]:
    """List ``(img, mask, skel)`` npy triples in a data directory.

    Args:
        data_dir: Directory of ``<name>_img.npy`` / ``_mask.npy`` / ``_skel.npy``.

    Returns:
        Sorted list of ``(img_path, mask_path, skel_path)`` tuples. ``skel_path``
        may be an empty string if no skeleton file exists.
    """
    triples = []
    for img_path in sorted(glob.glob(os.path.join(data_dir, "*_img.npy"))):
        stem = img_path[: -len("_img.npy")]
        mask_path = stem + "_mask.npy"
        skel_path = stem + "_skel.npy"
        if os.path.exists(mask_path):
            triples.append(
                (img_path, mask_path, skel_path if os.path.exists(skel_path) else "")
            )
    return triples


class NeuriteDataset(Dataset):
    """Random-crop, augmented dataset over rasterized neurite fields."""

    def __init__(
        self,
        pairs: List[Tuple[str, str, str]],
        crop: int = 256,
        length: int = 100,
        train: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            pairs: ``(img, mask, skel)`` path triples (from :func:`list_pairs`),
                already split into the desired train or val subset.
            crop: Square crop size in pixels.
            length: Number of samples per epoch (train mode). Ignored for the
                count in val mode, where one deterministic centre crop per field
                is returned.
            train: If True, random crop + augmentation; else fixed centre crop.
            seed: Optional RNG seed for reproducibility.
        """
        if not pairs:
            raise ValueError("NeuriteDataset received no (image, mask) pairs")
        self.pairs = pairs
        self.crop = crop
        self.length = length
        self.train = train
        self.rng = np.random.default_rng(seed)

        # Preload + normalize (data is small: a few 2048x2048 fields).
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        for img_path, mask_path, _ in pairs:
            self.images.append(percentile_normalize(np.load(img_path)))
            self.masks.append(np.load(mask_path).astype(np.float32))

    def __len__(self) -> int:
        """Return the epoch length.

        Returns:
            ``length`` in train mode, else one crop per field.
        """
        return self.length if self.train else len(self.pairs)

    def _random_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random (foreground-biased) square crop.

        Args:
            image: Normalized image (H, W).
            mask: Binary mask (H, W).

        Returns:
            ``(image_crop, mask_crop)`` of size ``crop`` x ``crop``.
        """
        h, w = image.shape
        c = self.crop
        if h <= c or w <= c:
            return self._pad_to_crop(image, mask)
        # Bias toward foreground: with prob 0.7 centre the crop on a mask pixel.
        fg = np.argwhere(mask > 0)
        if len(fg) and self.rng.random() < 0.7:
            cy, cx = fg[self.rng.integers(len(fg))]
            top = int(np.clip(cy - c // 2, 0, h - c))
            left = int(np.clip(cx - c // 2, 0, w - c))
        else:
            top = int(self.rng.integers(0, h - c + 1))
            left = int(self.rng.integers(0, w - c + 1))
        return image[top:top + c, left:left + c], mask[top:top + c, left:left + c]

    def _center_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic centre crop for validation.

        Args:
            image: Normalized image (H, W).
            mask: Binary mask (H, W).

        Returns:
            ``(image_crop, mask_crop)`` of size ``crop`` x ``crop``.
        """
        h, w = image.shape
        c = self.crop
        if h < c or w < c:
            return self._pad_to_crop(image, mask)
        top = (h - c) // 2
        left = (w - c) // 2
        return image[top:top + c, left:left + c], mask[top:top + c, left:left + c]

    def _pad_to_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-pad an image/mask up to the crop size.

        Args:
            image: Normalized image (H, W).
            mask: Binary mask (H, W).

        Returns:
            ``(image_crop, mask_crop)`` padded to at least ``crop`` x ``crop``
            then centre-cropped.
        """
        c = self.crop
        h, w = image.shape
        ph, pw = max(0, c - h), max(0, c - w)
        image = np.pad(image, ((0, ph), (0, pw)))
        mask = np.pad(mask, ((0, ph), (0, pw)))
        return image[:c, :c], mask[:c, :c]

    def _augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply label-preserving flips / 90-degree rotations / small jitter.

        Args:
            image: Crop image (H, W).
            mask: Crop mask (H, W).

        Returns:
            Augmented ``(image, mask)``.
        """
        k = int(self.rng.integers(0, 4))
        image, mask = np.rot90(image, k), np.rot90(mask, k)
        if self.rng.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if self.rng.random() < 0.5:
            image, mask = np.flipud(image), np.flipud(mask)
        # Mild intensity jitter on the image only (mask unchanged).
        if self.rng.random() < 0.5:
            gain = 1.0 + float(self.rng.uniform(-0.1, 0.1))
            bias = float(self.rng.uniform(-0.05, 0.05))
            image = np.clip(image * gain + bias, 0, 1)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one ``(image, mask)`` tensor pair.

        Args:
            idx: Sample index. In train mode it selects nothing deterministic
                (a random field is drawn); in val mode it selects field ``idx``.

        Returns:
            ``(image, mask)`` float32 tensors of shape (1, crop, crop).
        """
        if self.train:
            fi = int(self.rng.integers(0, len(self.images)))
            image, mask = self._random_crop(self.images[fi], self.masks[fi])
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._center_crop(self.images[idx], self.masks[idx])

        image_t = torch.from_numpy(image.astype(np.float32))[None]
        mask_t = torch.from_numpy((mask > 0).astype(np.float32))[None]
        return image_t, mask_t


if __name__ == "__main__":
    import sys

    data = sys.argv[1] if len(sys.argv) > 1 else "ml/neurite/data"
    pairs = list_pairs(data)
    print(f"found {len(pairs)} pair(s) in {data}")
    if pairs:
        ds = NeuriteDataset(pairs, crop=128, length=8, train=True, seed=0)
        img, msk = ds[0]
        print(f"sample img {tuple(img.shape)} range[{img.min():.2f},{img.max():.2f}] "
              f"mask fg-frac={msk.mean().item():.4f}")
        print("dataset.py smoke test PASSED")
