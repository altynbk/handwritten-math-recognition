"""Re-evaluate saved models without training or modifying published artifacts.

    python -m src.evaluate
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score

from .artifacts import load_metadata, load_saved_model
from .augment import augment_once
from .config import DATA_DIR, MODELS_DIR, RESULTS_DIR, MODEL_FILES, package_versions
from .data import dataset_digest, load_dataset, split_data


def hog_features(images):
    return np.array([hog(im.squeeze(), pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2)) for im in images])


def scores(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def evaluate_models(models_dir, X_test, y_test, X_augmented):
    """Return aggregate metrics and predictions on the same two test sets."""
    metrics, predictions = {}, {}
    for name in MODEL_FILES:
        model = load_saved_model(models_dir, name)
        metrics[name], predictions[name] = {}, {}
        for split, images in (("clean_test", X_test), ("augmented_test", X_augmented)):
            pred = (model.predict(hog_features(images)) if name == "SVM (HOG)"
                    else model.predict(images, verbose=0).argmax(axis=1))
            predictions[name][split] = pred
            metrics[name][split] = scores(y_test, pred)
    return metrics, predictions


def metric_differences(actual, expected, tolerance=1e-10):
    differences = []
    if actual.keys() != expected.keys():
        return ["Model names differ from the reference metrics."]
    for model in actual:
        for split in ("clean_test", "augmented_test"):
            for metric, value in actual[model][split].items():
                ref = expected[model].get(split, {}).get(metric)
                if ref is None or not np.isclose(value, ref, atol=tolerance, rtol=0):
                    differences.append(f"{model}/{split}/{metric}: {value} != {ref}")
    return differences


def evaluate_run(data_dir=DATA_DIR, models_dir=MODELS_DIR, expected_path=RESULTS_DIR / "metrics.json"):
    meta = load_metadata(models_dir)
    X, y, names, stats = load_dataset(data_dir, size=tuple(meta["img_size"]))
    if names != meta["class_names"] or stats != meta["dataset"]:
        raise ValueError("Dataset classes or counts differ from the saved run metadata.")
    parts = split_data(X, y, seed=meta["seed"])
    split_sizes = {k: len(v[0]) for k, v in parts.items()}
    if any(split_sizes[k] != meta[f"n_{k}"] for k in parts):
        raise ValueError("Split sizes differ from the saved run metadata.")
    fingerprints = {k: dataset_digest(*v, names) for k, v in parts.items()}
    if "split_sha256" in meta and fingerprints != meta["split_sha256"]:
        raise ValueError("Split fingerprints differ from the saved run metadata.")
    X_test, y_test = parts["test"]
    X_aug = augment_once(X_test, seed=meta["seed"] + 1)
    metrics, predictions = evaluate_models(models_dir, X_test, y_test, X_aug)
    expected = json.loads(Path(expected_path).read_text())
    differences = metric_differences(metrics, expected)
    pred = predictions["CNN (augmented)"]["clean_test"]
    errors = [{"test_index": int(i), "true": names[y_test[i]], "predicted": names[pred[i]]}
              for i in np.flatnonzero(pred != y_test)]
    return {"status": "passed" if not differences else "failed", "metrics": metrics,
            "differences": differences, "dataset": stats, "split_sizes": split_sizes,
            "split_sha256": fingerprints, "environment": package_versions(),
            "model_sha256": {name: hashlib.sha256((Path(models_dir) / file).read_bytes()).hexdigest()
                             for name, file in MODEL_FILES.items()},
            "cnn_augmented_clean_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    parser.add_argument("--expected", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--output", type=Path, help="Optional new JSON report; existing files are never overwritten.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Report already exists: {args.output}. Choose a new path.")
    report = evaluate_run(args.data, args.models, args.expected)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(text)
    print(text)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
