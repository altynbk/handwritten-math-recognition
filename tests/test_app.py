import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import app
from src.data import preprocess
from src.config import ROOT


@pytest.mark.parametrize("source", ["Draw", "Upload"])
def test_missing_or_blank_selected_input_does_not_fall_back(source, symbols):
    symbol, _ = symbols
    for empty in (None, Image.new("RGB", (100, 100), "white")):
        draw, upload = (empty, symbol) if source == "Draw" else (symbol, empty)
        with pytest.raises(app.gr.Error):
            app.predict(source, draw, upload, "SVM (HOG)", False)


@pytest.mark.parametrize("source", ["Draw", "Upload"])
def test_selected_source_is_used_and_svm_has_no_fake_probability(source, symbols):
    vertical, horizontal = symbols
    label, preview, overlay, info = app.predict(source, {"composite": vertical}, horizontal, "SVM (HOG)", False)
    selected = vertical if source == "Draw" else horizontal
    expected = Image.fromarray((preprocess(selected).squeeze() * 255).astype("uint8")).resize((192, 192), Image.Resampling.NEAREST)
    np.testing.assert_array_equal(np.asarray(preview), np.asarray(expected))
    assert isinstance(label, str) and overlay is None
    assert app.out_label.postprocess(label).confidences is None
    assert "does not provide probabilities" in info
    assert "test acc 0.9639, F1 0.9640" in info


@pytest.mark.parametrize("name", ["CNN (real only)", "CNN (augmented)"])
def test_saved_cnn_prediction_and_gradcam(name, symbols):
    label, preview, overlay, _ = app.predict("Upload", None, symbols[0], name, False)
    assert len(label) == 15 and all(np.isfinite(list(label.values())))
    assert sum(label.values()) == pytest.approx(1, abs=1e-5)
    assert max(label, key=label.get) == "1"
    assert preview.size == (192, 192) and overlay.size == (192, 192)


def test_missing_artifact_is_a_user_facing_error(monkeypatch, tmp_path, symbols):
    monkeypatch.setattr(app, "_cache", {})
    monkeypatch.setattr(app, "MODELS_DIR", str(tmp_path))
    with pytest.raises(app.gr.Error, match="Restore the model files"):
        app.predict("Draw", symbols[0], None, "CNN (augmented)", False)


def test_unsupported_source_or_model_is_rejected(symbols):
    with pytest.raises(app.gr.Error, match="Draw or Upload"):
        app.predict("invalid", symbols[0], None, "SVM (HOG)", False)
    with pytest.raises(app.gr.Error, match="available models"):
        app.predict("Draw", symbols[0], None, "invalid", False)


@pytest.mark.parametrize("contents", [None, "invalid JSON", "[]", "{}",
    '{"SVM (HOG)": {"clean_test": {"accuracy": null, "f1_macro": 0.9}}}'])
def test_missing_or_invalid_optional_metrics_do_not_break_prediction(monkeypatch, tmp_path, symbols, contents):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    if contents is not None:
        (tmp_path / "metrics.json").write_text(contents)
    label, preview, _, info = app.predict("Upload", None, symbols[0], "SVM (HOG)", False)
    assert isinstance(label, str) and preview.size == (192, 192)
    assert "metrics are unavailable" in info and "test acc" not in info


@pytest.mark.parametrize("results", [None, "custom-results"])
def test_custom_models_require_their_own_results_directory(tmp_path, results):
    env = dict(os.environ, MODELS_DIR=str(tmp_path))
    env.pop("RESULTS_DIR", None)
    if results is not None:
        env["RESULTS_DIR"] = results
    subprocess.run([sys.executable, "-c", f"import app; assert app.RESULTS_DIR == {results!r}"],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
