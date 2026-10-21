import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from src.augment import augment_dataset, augment_once
from src.data import has_content, load_dataset, preprocess, split_data


@pytest.mark.parametrize("mode,color", [("RGB", "white"), ("RGB", "black"),
                                      ("L", 100), ("RGBA", (0, 0, 0, 0))])
def test_empty_inputs_are_rejected(mode, color):
    assert not has_content(Image.new(mode, (100, 100), color))


def test_polarity_transparency_and_aspect_ratio(symbols):
    image, _ = symbols
    inverted = ImageOps.invert(image)
    rgba = image.convert("RGBA")
    pixels = np.asarray(rgba).copy()
    pixels[..., 3] = np.where(np.asarray(image)[..., 0] == 255, 0, 255)
    transparent = Image.fromarray(pixels)
    canonical = preprocess(image)
    for other in (inverted, transparent):
        assert has_content(other)
        np.testing.assert_array_equal(canonical, preprocess(other))
    assert canonical.shape == (64, 64, 1) and canonical.dtype == np.float32
    rows, cols = np.where(canonical.squeeze() > 0.25)
    assert np.ptp(cols) < np.ptp(rows) / 4


def test_deduplication_removes_all_conflicting_copies(tmp_path, symbols):
    vertical, horizontal = symbols
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(); second.mkdir()
    vertical.save(first / "original.png")
    vertical.save(first / "duplicate.png")
    vertical.save(second / "conflicting.png")
    horizontal.save(first / "keep.png")
    diagonal = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(diagonal).line((30, 30, 170, 170), fill="black", width=12)
    diagonal.save(second / "keep.png")
    (second / "broken.png").write_bytes(b"invalid image")
    X, y, names, stats = load_dataset(tmp_path)
    assert names == ["a", "b"] and y.tolist() == [0, 1]
    assert len(X) == 2 and stats == {"files": 6, "unreadable": 1, "duplicates_removed": 2,
                                   "label_conflicts_removed": 1, "kept": 2}


def test_missing_or_unusable_data_explains_recovery(tmp_path):
    with pytest.raises(FileNotFoundError, match="restore"):
        load_dataset(tmp_path / "missing")
    (tmp_path / "a").mkdir()
    with pytest.raises(ValueError, match="No usable"):
        load_dataset(tmp_path)


def test_splits_and_augmentation_are_reproducible_and_disjoint():
    X = np.random.default_rng(1).random((100, 64, 64, 1), dtype=np.float32)
    y = np.repeat([0, 1], 50)
    first = split_data(X, y, seed=42)
    second = split_data(X, y, seed=42)
    sets = []
    for name, (images, labels) in first.items():
        np.testing.assert_array_equal(images, second[name][0])
        np.testing.assert_array_equal(labels, second[name][1])
        sets.append({im.tobytes() for im in images})
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    train, labels = first["train"]
    extended, extended_y, _ = augment_dataset(train, labels, per_image=1, seed=42)
    np.testing.assert_array_equal(extended[:len(train)], train)
    np.testing.assert_array_equal(extended_y, np.tile(labels, 2))
    noisy = augment_once(first["test"][0], seed=43)
    np.testing.assert_array_equal(noisy, augment_once(first["test"][0], seed=43))
    assert np.isfinite(noisy).all() and 0 <= noisy.min() <= noisy.max() <= 1
