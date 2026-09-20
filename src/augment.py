"""Training-time augmentations (all operate on (H, W, 1) float32 in [0, 1],
light strokes on a dark background).

Three families, each modelling a different real-world variation:
  * stroke      - rotation + thicker/thinner pen + slight blur
  * occlusion   - a random patch that hides part of the symbol (smudges, cut-offs)
  * photometric - brightness/contrast change + Gaussian noise + small shift
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Tuple

import cv2
import numpy as np


def _u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img.squeeze(), 0, 1) * 255).astype("uint8")


def _f32(a: np.ndarray) -> np.ndarray:
    return (a.astype("float32") / 255.0)[..., None]


def stroke_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    a = _u8(img)
    angle = float(rng.integers(-18, 19))
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    a = cv2.warpAffine(a, m, (w, h), borderValue=0)
    kernel = np.ones((2, 2), np.uint8)
    a = cv2.dilate(a, kernel) if rng.random() < 0.5 else cv2.erode(a, kernel)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return _f32(a)


def occlusion_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    a = _u8(img).copy()
    x1 = int(rng.integers(0, w - 15))
    y1 = int(rng.integers(0, h - 15))
    x2 = x1 + int(rng.integers(10, 20))
    y2 = y1 + int(rng.integers(10, 20))
    cv2.rectangle(a, (x1, y1), (x2, y2), int(rng.integers(0, 80)), -1)
    return _f32(a)


def photometric_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    a = _u8(img)
    a = cv2.convertScaleAbs(a, alpha=float(rng.uniform(0.8, 1.2)), beta=float(rng.integers(-15, 15)))
    noise = rng.normal(0, 5, (h, w)).astype("float32")
    a = np.clip(a.astype("float32") + noise, 0, 255).astype("uint8")
    m = np.float32([[1, 0, int(rng.integers(-2, 3))], [0, 1, int(rng.integers(-2, 3))]])
    a = cv2.warpAffine(a, m, (w, h), borderValue=0)
    return _f32(a)


AUGMENTATIONS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "stroke": stroke_aug,
    "occlusion": occlusion_aug,
    "photometric": photometric_aug,
}


def random_augment(img: np.ndarray, rng: np.random.Generator, max_tries: int = 5) -> Tuple[np.ndarray, str]:
    """Apply one random augmentation; retry if it destroys most of the strokes."""
    names = list(AUGMENTATIONS)
    for _ in range(max_tries):
        name = names[int(rng.integers(len(names)))]
        out = AUGMENTATIONS[name](img, rng)
        if out.sum() >= 0.3 * img.sum():
            return out, name
    return img.copy(), "none"


def augment_dataset(X: np.ndarray, y: np.ndarray, per_image: int = 2, seed: int = 0):
    """Return originals + `per_image` augmented copies of every image."""
    rng = np.random.default_rng(seed)
    xs, ys, counts = [X], [y], Counter()
    for _ in range(per_image):
        batch = []
        for img in X:
            out, name = random_augment(img, rng)
            batch.append(out)
            counts[name] += 1
        xs.append(np.stack(batch))
        ys.append(y)
    return np.concatenate(xs).astype("float32"), np.concatenate(ys), dict(counts)


def augment_once(X: np.ndarray, seed: int = 123) -> np.ndarray:
    """One fixed-seed augmented copy of X (used as a 'harder' test set)."""
    rng = np.random.default_rng(seed)
    return np.stack([random_augment(img, rng)[0] for img in X]).astype("float32")
