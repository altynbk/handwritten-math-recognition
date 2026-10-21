"""Shared locations and the settings of the published experiment."""
from pathlib import Path
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "handwritten_dataset"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEED = 42
EPOCHS = 12
AUG_PER_IMAGE = 2
MODEL_FILES = {
    "SVM (HOG)": "svm_hog.joblib",
    "CNN (real only)": "cnn_real.keras",
    "CNN (augmented)": "cnn_aug.keras",
}


def package_versions():
    return {name: version(name) for name in (
        "numpy", "pillow", "opencv-python-headless", "scikit-learn",
        "scikit-image", "tensorflow", "keras", "joblib", "gradio",
    )}
