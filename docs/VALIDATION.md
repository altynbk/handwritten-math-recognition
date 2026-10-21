# Validation and reproducibility

The included model weights, dataset and published result files are unchanged.
This maintenance pass makes the existing experiment runnable and inspectable;
it does not claim improved model accuracy.

## Saved-model verification

On 2026-10-21, Python 3.12 on macOS arm64 reproduced all 12 published metrics:
accuracy and macro F1 for all three models on the clean and augmented test views.
The absolute comparison tolerance is `1e-10`, with no relative tolerance.
The dataset counts and split sizes also match the original metadata.

[Machine-readable report](verification-macos.json) records the metrics, dependency
versions, ordered split fingerprints, model checksums and the nine clean-test
errors of the augmented CNN. Run `python -m src.evaluate` to check independently.
The original training environment was not recorded, so this is a verified
inference environment, not a reconstruction of the original training machine.

Two dependency details matter:

- The SVM artifact was saved with scikit-learn 1.8.0. That version is pinned;
  regression tests treat incompatible-version loading warnings as errors.
- OpenCV 5.0.0.93 changed the observed SVM augmented macro F1 to
  `0.8461449382365633`, while the published value is `0.8465519335274385`.
  OpenCV 4.14.0.94 reproduces the published value. Accuracy alone did not reveal
  the discrepancy. The evaluator checks both metrics, and the working version
  is pinned rather than loosening the comparison or editing published results.

## Checks

- 35 tests cover empty inputs, source selection, real saved-model
  inference, SVM output without invented probabilities, Grad-CAM, preprocessing,
  deterministic splitting and augmentation, duplicate/conflicting-label
  removal, artifact errors, and protection against overwriting published runs.
  They also check that missing or invalid metrics fail verification, custom
  models cannot inherit the bundled benchmark scores, and optional display
  metrics cannot break inference.
- All six code cells of `notebooks/walkthrough.ipynb` execute without training.
  Committed outputs show the actual saved-model results and error examples.
- An automated training test fits all three models on 60 synthetic images in
  two classes (one epoch per CNN), writes plots and metadata, and reloads the
  artifacts to reproduce its own metrics. This runs in the standard test suite;
  its scores are not project benchmark results.
- Browser checks covered drawing, undo/redo, source switching with both inputs
  populated, upload, empty upload rejection, clearing stale results, all three
  models, random augmentation and Grad-CAM. Switching inputs during inference
  cancels the old response so it cannot repopulate the cleared outputs.
  Gradio 6.30.0 fixes the observed
  canvas history failure in 5.49.1/5.50.0. No frontend errors occurred in the
  final demo session.
- GitHub Actions runs dependency checks, regression tests and saved-model
  evaluation on Linux and macOS. Linux also executes the notebook. Each run
  preserves verification JSON and, on Linux, the executed notebook as downloadable
  artifacts for 14 days. PRs are checked once per update; pushes to `main` and
  manual runs are also supported.
- A fresh Python 3.12 environment with only `requirements.txt` installed runs
  the demo and reproduces all published metrics, without test or notebook packages.

## Updating dependencies

Use Python 3.12. Direct dependencies live in `requirements.in`,
`requirements-dev.in` and `requirements-notebook.in`. Their corresponding `.txt`
files pin the full resolution, including platform-specific dependencies.
To deliberately regenerate the locks with uv:

```bash
uv pip compile --python-version 3.12 --universal requirements.in -o requirements.txt
uv pip compile --python-version 3.12 --universal -c requirements.txt requirements-dev.in -o requirements-dev.txt
uv pip compile --python-version 3.12 --universal -c requirements.txt requirements-notebook.in -o requirements-notebook.txt
```

Install into a fresh environment and rerun the checks before accepting an
update. Fixing the seed does not guarantee bit-identical retraining across
hardware or library versions. New training runs save their environment and split
fingerprints so that later differences can be investigated.

For the experiment's statistical and data limitations, see
[dataset and protocol notes](DATASET.md).
