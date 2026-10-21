import copy
import json

import pytest

from src.artifacts import load_metadata
from src.config import ROOT
from src.evaluate import metric_differences
from src.train import prepare_output_dirs, run


def test_published_artifacts_are_protected_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="protected"):
        prepare_output_dirs(ROOT / "models", tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_existing_run_is_not_overwritten(tmp_path):
    previous = tmp_path / "results"
    previous.mkdir()
    report = previous / "metrics.json"
    report.write_text("original")
    with pytest.raises(FileExistsError):
        prepare_output_dirs(tmp_path / "models", previous)
    assert report.read_text() == "original" and not (tmp_path / "models").exists()


def test_overlapping_destinations_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-overlapping"):
        prepare_output_dirs(tmp_path, tmp_path / "results")


def test_invalid_training_parameters_do_not_create_a_run(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run(out_dir=tmp_path / "models", results_dir=tmp_path / "results", epochs=0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("metadata", [[], {"class_names": ["1", "1"], "img_size": [64, 64]},
    {"class_names": [""], "img_size": [64, 64]}, {"class_names": ["1"], "img_size": [True, 64]}])
def test_metadata_validation(tmp_path, metadata):
    (tmp_path / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Invalid"):
        load_metadata(tmp_path)


def test_verification_detects_changed_and_missing_results():
    expected = json.loads((ROOT / "results/metrics.json").read_text())
    changed = copy.deepcopy(expected)
    changed["SVM (HOG)"]["augmented_test"]["f1_macro"] -= 0.0004
    differences = metric_differences(changed, expected)
    assert len(differences) == 1 and "f1_macro" in differences[0]
    del changed["SVM (HOG)"]
    assert metric_differences(changed, expected)


@pytest.mark.parametrize("side", ["actual", "expected"])
def test_verification_rejects_incomplete_and_invalid_reports(side):
    reference = json.loads((ROOT / "results/metrics.json").read_text())
    for invalid in (None, float("nan"), float("inf"), "0.8", True, -0.1, 1.1):
        reports = {"actual": copy.deepcopy(reference), "expected": copy.deepcopy(reference)}
        reports[side]["SVM (HOG)"]["augmented_test"]["f1_macro"] = invalid
        assert metric_differences(**reports)
    for missing in ("metric", "split"):
        reports = {"actual": copy.deepcopy(reference), "expected": copy.deepcopy(reference)}
        model = reports[side]["SVM (HOG)"]
        if missing == "metric":
            del model["augmented_test"]["f1_macro"]
        else:
            del model["augmented_test"]
        assert metric_differences(**reports)
    assert metric_differences({}, {})
