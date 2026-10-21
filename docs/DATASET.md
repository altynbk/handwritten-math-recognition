# Dataset and evaluation scope

The bundled dataset contains 8,384 PNG/JPEG images in 15 class folders. The
project team collected the drawings themselves, as confirmed by the author.
Writer identities, collection dates and a collection protocol were not recorded
in the original metadata; no writer count or writer-independent evaluation is
claimed here.

Most images are approximately 155×135 pixels, with some 400×400 images.
The classes are digits `0`–`9` and `add`, `sub`, `mul`, `div`, `eq`.
The repository carries an MIT license. No external dataset is used.

## Preparation

`src/data.py` composites transparency on white, converts to grayscale, normalizes
polarity from the image border, crops to the symbol, pads to preserve its aspect
ratio and resizes to 64×64. Training and inference share this function.

Duplicate detection hashes the canonical pixels rounded to 8-bit values. It
removes 347 repeated occurrences and the remaining copy of one image that had
conflicting labels. No input files are changed. All 8,384 files are readable;
8,036 unambiguous, unique canonical images remain.

The stratified split uses seed 42:

| Partition | Images | Use |
|---|---:|---|
| Train | 5,785 | Fit SVM and CNN weights; generate training augmentations |
| Validation | 643 | Early stopping and learning-rate scheduling |
| Test | 1,608 | Report accuracy, macro F1, confusion matrix and errors |

The test proportion is 20% of the cleaned dataset. Validation is 10% of the
remaining 80%, so the final proportions are approximately 72% / 8% / 20%.
An additional test view applies one augmentation per held-out image with seed 43.
This is a transformation of the same 1,608 test images, not an independent sample.

## Limits

- Exact deduplication does not remove all near-duplicates or shared writing styles.
- No writer or session grouping is available. High image-level accuracy does not
  establish performance on new people, arbitrary photographs or a new dataset.
- The perturbation test uses the same augmentation families as training.
- The augmented CNN receives roughly three times as many gradient updates as the
  real-only CNN. The experiment does not isolate augmentation from compute budget.
- There is one split and one training seed. The one-error difference between the
  CNNs on the clean test does not establish a general ranking.

The walkthrough displays the augmented CNN's nine clean-test errors, including
confusions between `0` and `6`, and between `div` and `eq`. These are limitations
of the existing classifier, not examples removed from evaluation.
