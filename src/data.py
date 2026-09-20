"""Dataset loading, polarity normalisation and preprocessing."""
from __future__ import annotations

import hashlib
import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (64, 64)
IMG_EXTS = (".png", ".jpg", ".jpeg")


def to_gray_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 grayscale array in [0, 1] (transparency -> white)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def normalize_polarity(arr: np.ndarray) -> np.ndarray:
    """Make every image 'light strokes on dark background'.

    Decided per image from the median of the border pixels, so black-on-white
    and white-on-black inputs (dataset files, uploads, canvas drawings) all end
    up in the same canonical form.
    """
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(border) > 0.5:
        arr = 1.0 - arr
    return arr


def crop_to_content(arr: np.ndarray, margin: float = 0.15, thresh: float = 0.25) -> np.ndarray:
    """Crop to the bounding box of the strokes and pad to a square."""
    ys, xs = np.where(arr > thresh)
    if len(ys) == 0:
        return arr
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    side = int(max(h, w) * (1 + 2 * margin))
    canvas = np.zeros((side, side), dtype=np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = arr[y0:y1 + 1, x0:x1 + 1]
    return canvas


def preprocess(img: Image.Image, size: Tuple[int, int] = IMG_SIZE, autocrop: bool = True) -> np.ndarray:
    """PIL image -> (H, W, 1) float32 array in [0, 1].

    Canonical form: light strokes on dark background, cropped to the symbol's
    bounding box (padded to a square, aspect ratio preserved), resized to `size`.
    Training and the demo use exactly this function.
    """
    arr = normalize_polarity(to_gray_array(img))
    if autocrop:
        arr = crop_to_content(arr)
    arr = cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return arr.astype("float32")[..., None]


def load_dataset(data_dir: str, size: Tuple[int, int] = IMG_SIZE):
    """Load `data_dir/<class_name>/*.png|jpg`.

    Returns X (N,H,W,1), y (N,), class_names, stats.

    Exact duplicates (identical preprocessed pixels) are dropped *before* any
    split, so the same drawing cannot end up in both train and test. If the same
    drawing appears under two different labels, all copies are removed.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            "Run `python scripts/download_data.py --url <dataset zip url>` first."
        )
    class_names: List[str] = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if not class_names:
        raise RuntimeError(f"No class sub-folders found in {data_dir}")

    seen = {}          # pixel hash -> index into X
    conflicts = set()  # hashes that occur under more than one label
    X, y = [], []
    stats = {"files": 0, "unreadable": 0, "duplicates_removed": 0, "label_conflicts_removed": 0}
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            stats["files"] += 1
            try:
                with Image.open(os.path.join(cls_dir, fname)) as im:
                    arr = preprocess(im, size)
            except Exception:
                stats["unreadable"] += 1
                continue
            key = hashlib.md5((arr * 255).round().astype("uint8").tobytes()).hexdigest()
            if key in seen:
                stats["duplicates_removed"] += 1
                if y[seen[key]] != idx:
                    conflicts.add(key)
                continue
            seen[key] = len(X)
            X.append(arr)
            y.append(idx)

    if conflicts:
        keep = [i for i, k in enumerate(seen) if k not in conflicts]
        stats["label_conflicts_removed"] = len(seen) - len(keep)
        X = [X[i] for i in keep]
        y = [y[i] for i in keep]
    stats["kept"] = len(X)
    return np.stack(X).astype("float32"), np.array(y, dtype="int64"), class_names, stats


def split_data(X, y, seed: int = 42, test_size: float = 0.2, val_size: float = 0.1):
    """Stratified train / val / test split.

    * test  - touched exactly once, for the final numbers
    * val   - used for EarlyStopping / LR scheduling
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val_size, random_state=seed, stratify=y_tr)
    return {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)}
