"""Load the trusted local artifacts used by the demo and evaluation."""
import json
from pathlib import Path

from joblib import load

from .config import MODEL_FILES


def load_metadata(directory):
    path = Path(directory) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Restore the model files from the repository.")
    with path.open() as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid metadata object in {path}.")
    names, size = meta.get("class_names"), meta.get("img_size")
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) and n.strip() for n in names) or len(set(names)) != len(names)
            or not isinstance(size, list) or len(size) != 2
            or not all(type(n) is int and n > 0 for n in size)):
        raise ValueError(f"Invalid class names or image size in {path}.")
    return meta


def load_saved_model(directory, name):
    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}")
    path = Path(directory) / MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Restore the model files from the repository.")
    if name == "SVM (HOG)":
        return load(path)
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)
