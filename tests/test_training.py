"""Exercise the real train -> save -> reload -> evaluate path on tiny fixture data."""
import json

import numpy as np
from PIL import Image, ImageDraw

from src.evaluate import evaluate_run
from src.train import run


def test_training_round_trip(tmp_path):
    rng = np.random.default_rng(10)
    for label in ("horizontal", "vertical"):
        folder = tmp_path / "data" / label
        folder.mkdir(parents=True)
        for index in range(30):
            image = Image.new("L", (96, 96), 255)
            points = list(zip(rng.integers(35, 60, 5).tolist(),
                              np.linspace(12, 83, 5).astype(int).tolist()))
            if label == "horizontal":
                points = [(y, x) for x, y in points]
            ImageDraw.Draw(image).line(points, fill=0, width=int(rng.integers(4, 9)))
            image.save(folder / f"{index:02d}.png")

    data, models, results = (tmp_path / name for name in ("data", "models", "results"))
    trained_metrics = run(data, models, results, epochs=1, aug_per_image=1)
    report = evaluate_run(data, models, results / "metrics.json")
    assert report["status"] == "passed", report["differences"]
    assert report["metrics"] == trained_metrics
    assert report["dataset"]["kept"] == 60
    meta = json.loads((models / "meta.json").read_text())
    assert meta["split_sha256"] == report["split_sha256"]
    assert meta["environment"] == report["environment"]
    assert set(meta["epochs_run"].values()) == {1}
    for filename in ("augmentation_examples.png", "confusion_matrix.png",
                     "gradcam_examples.png", "comparison.png", "results.md"):
        assert (results / filename).stat().st_size > 0
