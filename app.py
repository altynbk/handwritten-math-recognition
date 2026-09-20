"""Gradio demo: draw or upload a symbol, pick a model, see prediction + Grad-CAM.

    python app.py
"""
from __future__ import annotations

import json
import os

import gradio as gr
import numpy as np
from joblib import load
from PIL import Image
from skimage.feature import hog

from src.augment import random_augment
from src.data import preprocess, to_gray_array
from src.models import gradcam_heatmap, overlay_heatmap

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

CHOICES = ["CNN (augmented)", "CNN (real only)", "SVM (HOG)"]
_cache: dict = {}


def _meta() -> dict:
    path = os.path.join(MODELS_DIR, "meta.json")
    if not os.path.exists(path):
        raise gr.Error("Models not found. Train first: `python -m src.train`.")
    if "meta" not in _cache:
        with open(path) as f:
            _cache["meta"] = json.load(f)
    return _cache["meta"]


def _metrics() -> dict:
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _model(name: str):
    if name in _cache:
        return _cache[name]
    fname = {"CNN (augmented)": "cnn_aug.keras", "CNN (real only)": "cnn_real.keras",
             "SVM (HOG)": "svm_hog.joblib"}[name]
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise gr.Error(f"Missing model file: {path}. Train first: `python -m src.train`.")
    if name == "SVM (HOG)":
        _cache[name] = load(path)
    else:
        import tensorflow as tf  # imported lazily so the UI starts fast
        _cache[name] = tf.keras.models.load_model(path)
    return _cache[name]


def _to_pil(value):
    """Accept PIL / ndarray / Gradio editor dict; return a PIL image or None."""
    if value is None:
        return None
    if isinstance(value, dict):  # ImageEditor / Sketchpad value
        value = value.get("composite")
        if value is None:
            return None
    if isinstance(value, np.ndarray):
        value = Image.fromarray(value.astype("uint8"))
    return value


def _has_ink(img: Image.Image) -> bool:
    """True if a canvas image contains dark strokes (transparent -> white)."""
    return bool(to_gray_array(img).min() < 0.8)


def predict(sketch, upload, model_name, add_noise):
    img = _to_pil(sketch)
    if img is None or not _has_ink(img):
        img = _to_pil(upload)
    if img is None:
        raise gr.Error("Draw a symbol or upload an image first.")

    meta = _meta()
    class_names = meta["class_names"]
    x = preprocess(img, tuple(meta["img_size"]))
    if add_noise:
        x, _ = random_augment(x, np.random.default_rng())

    model = _model(model_name)
    overlay = None
    if model_name == "SVM (HOG)":
        pred = int(model.predict(hog(x.squeeze(), pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2))[None])[0])
        label = {class_names[pred]: 1.0}  # SVM gives a class, not probabilities
        note = "SVM outputs a class only - no confidence scores."
    else:
        probs = model.predict(x[None], verbose=0)[0]
        pred = int(np.argmax(probs))
        label = {c: float(p) for c, p in zip(class_names, probs)}
        overlay = Image.fromarray(overlay_heatmap(x, gradcam_heatmap(model, x, pred)))
        note = ""

    m = _metrics().get(model_name)
    info = (f"{model_name} | test acc {m['clean_test']['accuracy']:.4f}, "
            f"F1 {m['clean_test']['f1_macro']:.4f}" if m else model_name)
    if note:
        info += f"\n{note}"

    shown = Image.fromarray((x.squeeze() * 255).astype("uint8")).resize((192, 192), Image.NEAREST)
    return label, shown, overlay, info


with gr.Blocks(title="Handwritten digits & math symbols") as demo:
    gr.Markdown("## Handwritten digits & math symbols\n"
                "15 classes: `0-9`, `add`, `sub`, `mul`, `div`, `eq`.")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Draw"):
                    sketch = gr.Sketchpad(type="pil", label="Draw a symbol", canvas_size=(280, 280),
                                          brush=gr.Brush(default_size=10, colors=["#000000"], color_mode="fixed"))
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", sources=["upload", "clipboard"], label="Upload an image")
            model_sel = gr.Radio(CHOICES, value=CHOICES[0], label="Model")
            noise = gr.Checkbox(False, label="Apply a random augmentation before predicting")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Prediction")
            with gr.Row():
                out_img = gr.Image(label="Model input (64x64)", interactive=False)
                out_cam = gr.Image(label="Grad-CAM (CNN only)", interactive=False)
            out_info = gr.Textbox(label="Model info", lines=2)
    btn.click(predict, [sketch, upload, model_sel, noise], [out_label, out_img, out_cam, out_info])

if __name__ == "__main__":
    demo.launch()
