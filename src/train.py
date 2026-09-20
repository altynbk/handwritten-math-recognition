"""Train and evaluate HOG+SVM, CNN (real data only) and CNN (real + augmented).

    python -m src.train --data data/handwritten_dataset

Fair comparison: both CNNs share architecture, callbacks and validation split.
The test split is used once, at the end. Models are additionally evaluated on a
fixed-seed *augmented* copy of the test set to measure robustness.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import dump
from skimage.feature import hog
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from .augment import AUGMENTATIONS, augment_dataset, augment_once
from .data import IMG_SIZE, load_dataset, split_data
from .models import build_cnn, gradcam_heatmap, make_callbacks, overlay_heatmap


def hog_features(X: np.ndarray) -> np.ndarray:
    return np.array([hog(im.squeeze(), pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for im in X])


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_aug_examples(X, y, class_names, path, seed):
    rng = np.random.default_rng(seed)
    n_cls = min(4, len(class_names))
    fig, axes = plt.subplots(n_cls, 1 + len(AUGMENTATIONS) * 2, figsize=(12, 2 * n_cls))
    for r in range(n_cls):
        img = X[np.where(y == r)[0][0]]
        axes[r, 0].imshow(img.squeeze(), cmap="gray")
        axes[r, 0].set_title(f"{class_names[r]} (orig)", fontsize=8)
        c = 1
        for name, fn in AUGMENTATIONS.items():
            for _ in range(2):
                axes[r, c].imshow(fn(img, rng).squeeze(), cmap="gray")
                axes[r, c].set_title(name, fontsize=8)
                c += 1
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gradcam_examples(model, X, y_true, class_names, path, n=6):
    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    fig, axes = plt.subplots(2, len(idx), figsize=(2 * len(idx), 4.4))
    for j, i in enumerate(idx):
        pred = int(np.argmax(model.predict(X[i][None], verbose=0)[0]))
        heat = gradcam_heatmap(model, X[i], pred)
        axes[0, j].imshow(X[i].squeeze(), cmap="gray")
        axes[0, j].set_title(f"true {class_names[y_true[i]]}", fontsize=8)
        axes[1, j].imshow(overlay_heatmap(X[i], heat))
        axes[1, j].set_title(f"pred {class_names[pred]}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_comparison(metrics, path):
    names = list(metrics)
    clean = [metrics[n]["clean_test"]["accuracy"] for n in names]
    noisy = [metrics[n]["augmented_test"]["accuracy"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - 0.2, clean, 0.4, label="clean test")
    ax.bar(x + 0.2, noisy, 0.4, label="augmented test")
    ax.set_xticks(x, names)
    ax.set_ylabel("accuracy")
    ax.set_ylim(max(0.0, min(clean + noisy) - 0.05), 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(data_dir="data/handwritten_dataset", out_dir="models", results_dir="results",
        epochs=30, aug_per_image=2, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    X, y, class_names, data_stats = load_dataset(data_dir)
    print("dataset:", data_stats)
    parts = split_data(X, y, seed=seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = parts["train"], parts["val"], parts["test"]
    print(f"classes={len(class_names)} train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # augmentation is applied to the training split only
    X_ext, y_ext, aug_counts = augment_dataset(X_tr, y_tr, per_image=aug_per_image, seed=seed)
    print(f"extended train: {len(X_ext)} (aug counts: {aug_counts})")
    X_te_aug = augment_once(X_te, seed=seed + 1)
    save_aug_examples(X_tr, y_tr, class_names, os.path.join(results_dir, "augmentation_examples.png"), seed)

    metrics = {}

    # ---- HOG + linear SVM --------------------------------------------------
    svm = LinearSVC(dual="auto").fit(hog_features(X_tr), y_tr)
    metrics["SVM (HOG)"] = {
        "clean_test": scores(y_te, svm.predict(hog_features(X_te))),
        "augmented_test": scores(y_te, svm.predict(hog_features(X_te_aug))),
    }
    dump(svm, os.path.join(out_dir, "svm_hog.joblib"))

    # ---- CNN (real only) and CNN (real + augmented) ---------------------------
    models, epochs_run = {}, {}
    for tag, (Xf, yf), fname in [
        ("CNN (real only)", (X_tr, y_tr), "cnn_real.keras"),
        ("CNN (augmented)", (X_ext, y_ext), "cnn_aug.keras"),
    ]:
        model = build_cnn(len(class_names), X_tr.shape[1:])
        hist = model.fit(Xf, yf, validation_data=(X_va, y_va), epochs=epochs, batch_size=64,
                         callbacks=make_callbacks(), verbose=2)
        epochs_run[tag] = len(hist.history["loss"])
        metrics[tag] = {
            "clean_test": scores(y_te, np.argmax(model.predict(X_te, verbose=0), 1)),
            "augmented_test": scores(y_te, np.argmax(model.predict(X_te_aug, verbose=0), 1)),
        }
        models[tag] = model
        model.save(os.path.join(out_dir, fname))

    # ---- artefacts ----------------------------------------------------------
    best = models["CNN (augmented)"]
    cm = confusion_matrix(y_te, np.argmax(best.predict(X_te, verbose=0), 1))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix - CNN (augmented), clean test")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    save_gradcam_examples(best, X_te, y_te, class_names, os.path.join(results_dir, "gradcam_examples.png"))
    save_comparison(metrics, os.path.join(results_dir, "comparison.png"))

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"class_names": class_names, "img_size": list(IMG_SIZE), "seed": seed,
                   "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
                   "aug_per_image": aug_per_image, "aug_counts": aug_counts,
                   "max_epochs": epochs, "epochs_run": epochs_run, "dataset": data_stats}, f, indent=2)

    lines = ["| Model | Clean test acc | Clean test F1 | Augmented test acc | Augmented test F1 |",
             "|---|---|---|---|---|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['clean_test']['accuracy']:.4f} | {m['clean_test']['f1_macro']:.4f} | "
                     f"{m['augmented_test']['accuracy']:.4f} | {m['augmented_test']['f1_macro']:.4f} |")
    table = "\n".join(lines)
    with open(os.path.join(results_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/handwritten_dataset")
    p.add_argument("--out", default="models")
    p.add_argument("--results", default="results")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--aug-per-image", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.data, a.out, a.results, a.epochs, a.aug_per_image, a.seed)


if __name__ == "__main__":
    main()
