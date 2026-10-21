"""Gradio demo: draw or upload a symbol, pick a model, see prediction + Grad-CAM.

    python app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.artifacts import load_metadata, load_saved_model
from src.augment import random_augment
from src.config import MODELS_DIR as DEFAULT_MODELS_DIR, RESULTS_DIR as DEFAULT_RESULTS_DIR
from src.data import has_content, preprocess
from src.evaluate import hog_features

MODELS_DIR = os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR))
# A different model directory must not inherit the bundled model's scores.
RESULTS_DIR = os.environ.get("RESULTS_DIR", str(DEFAULT_RESULTS_DIR)
                             if Path(MODELS_DIR).resolve() == DEFAULT_MODELS_DIR.resolve() else None)

CHOICES = ["CNN (augmented)", "CNN (real only)", "SVM (HOG)"]
_cache: dict = {}


def _meta() -> dict:
    if "meta" not in _cache:
        _cache["meta"] = load_metadata(MODELS_DIR)
    return _cache["meta"]


def _clean_metrics(model_name):
    """Optional display information must never prevent model inference."""
    if RESULTS_DIR is None:
        return None
    try:
        with (Path(RESULTS_DIR) / "metrics.json").open() as handle:
            metrics = json.load(handle)[model_name]["clean_test"]
        values = [metrics[key] for key in ("accuracy", "f1_macro")]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and np.isfinite(v) and 0 <= v <= 1 for v in values):
            return metrics
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _model(name: str):
    if name in _cache:
        return _cache[name]
    _cache[name] = load_saved_model(MODELS_DIR, name)
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


def predict(input_source, sketch, upload, model_name, add_noise):
    if input_source not in ("Draw", "Upload"):
        raise gr.Error("Choose Draw or Upload.")
    img = _to_pil(sketch if input_source == "Draw" else upload)
    if img is None:
        raise gr.Error("Draw a symbol first." if input_source == "Draw" else "Upload an image first.")
    if not has_content(img):
        raise gr.Error("The image is empty. Add a visible symbol and try again.")
    if model_name not in CHOICES:
        raise gr.Error("Choose one of the available models.")

    try:
        meta = _meta()
        model = _model(model_name)
    except (OSError, ValueError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc
    class_names = meta["class_names"]
    x = preprocess(img, tuple(meta["img_size"]))
    if add_noise:
        x, _ = random_augment(x, np.random.default_rng())

    overlay = None
    if model_name == "SVM (HOG)":
        pred = int(model.predict(hog_features(x[None]))[0])
        label = class_names[pred]
        note = "SVM outputs a class only; it does not provide probabilities."
    else:
        from src.models import gradcam_heatmap, overlay_heatmap
        probs = model.predict(x[None], verbose=0)[0]
        pred = int(np.argmax(probs))
        label = {c: float(p) for c, p in zip(class_names, probs)}
        overlay = Image.fromarray(overlay_heatmap(x, gradcam_heatmap(model, x, pred))).resize(
            (192, 192), Image.Resampling.NEAREST)
        note = f"Scores compare the {len(class_names)} known classes; they do not establish that an image is a supported symbol."

    m = _clean_metrics(model_name)
    if m:
        info = f"{model_name} | test acc {m['accuracy']:.4f}, F1 {m['f1_macro']:.4f}"
    else:
        info = f"{model_name}\nSaved evaluation metrics are unavailable."
    info += f"\n{note}"

    shown = Image.fromarray((x.squeeze() * 255).astype("uint8")).resize((192, 192), Image.Resampling.NEAREST)
    return label, shown, overlay, info


with gr.Blocks(title="Handwritten digits & math symbols") as demo:
    gr.Markdown("## Handwritten digits & math symbols\n"
                "Draw or upload **one symbol**. 15 classes: `0-9`, `add`, `sub`, `mul`, `div`, `eq`.")
    with gr.Row():
        with gr.Column():
            source = gr.Radio(["Draw", "Upload"], value="Draw", label="Input source")
            sketch = gr.Sketchpad(type="pil", label="Draw a symbol", canvas_size=(280, 280),
                                 brush=gr.Brush(default_size=10, colors=["#000000"], color_mode="fixed"))
            upload = gr.Image(type="pil", sources=["upload", "clipboard"],
                              label="Upload an image", visible=False)
            model_sel = gr.Radio(CHOICES, value=CHOICES[0], label="Model")
            noise = gr.Checkbox(False, label="Apply a random augmentation before predicting")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Prediction")
            with gr.Row():
                out_img = gr.Image(label="Model input (64x64)", interactive=False, height=220,
                                   buttons=["download", "fullscreen"])
                out_cam = gr.Image(label="Grad-CAM (CNN only)", interactive=False, height=220,
                                   buttons=["download", "fullscreen"])
            out_info = gr.Textbox(label="Model info", lines=3)
    outputs = [out_label, out_img, out_cam, out_info]
    prediction = btn.click(predict, [source, sketch, upload, model_sel, noise], outputs)
    # Changing an input also cancels the response of any in-flight prediction.
    source.change(lambda value: (gr.update(visible=value == "Draw"), gr.update(visible=value == "Upload"),
                                 None, None, None, ""),
                  source, [sketch, upload, *outputs], queue=False, cancels=[prediction])
    for component in (sketch, upload, model_sel, noise):
        component.change(lambda: (None, None, None, ""), outputs=outputs,
                         queue=False, cancels=[prediction])

if __name__ == "__main__":
    demo.launch()
