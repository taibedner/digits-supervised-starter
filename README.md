# Digits Supervised Starter

A small repository for **learning supervised image classification**. Train a model to recognize handwritten digits from **0 to 9**, inspect its mistakes, and predict one image at a time.

The code uses **scikit-learn, logistic regression, and a CPU**. The focus is understanding inputs, labels, train/validation/test splits, and evaluation before moving to larger neural networks.

**Execution status:** this repository was prepared without running installation, training, inference, or pytest. No measured accuracy or passing-test claim is included. The GitHub Actions workflow is manual and has not been started.

## What you will learn

- How an image becomes a feature vector.
- What the correct label means in supervised learning.
- Why training, validation, and test data must remain separate.
- How a model learns from `(image, label)` pairs.
- How to read accuracy, a confusion matrix, and incorrect predictions.
- How to save a model and use the same preprocessing during prediction.

## Dataset

The project uses scikit-learn's built-in [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset: **1,797 labeled grayscale images**, each **8 × 8 pixels**, with pixel intensities from **0 to 16**. Dataset loading does not require a separate download after scikit-learn is installed.

An image has shape `(8, 8)`. The model receives a flattened vector of **64 pixel values**. Its target is an integer from `0` through `9`.

This is a small handwritten-digit example. It does not recognize arbitrary objects, scenes, or text documents. Your own digit images can differ substantially from this dataset.

## Setup

Use Python 3.11 or newer. Python 3.12 is a straightforward starting point.

```bash
git clone https://github.com/taibedner/digits-supervised-starter.git
cd digits-supervised-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, use `python` instead of `python3` and activate with:

```powershell
.venv\Scripts\Activate.ps1
```

If you only want to use the program, `python -m pip install -e .` installs its runtime dependencies without pytest.

## 1. Look at the labeled images

```bash
digits-starter preview --output outputs/samples.png
```

Open `outputs/samples.png`. Each image shows its correct label. These labels are the supervision signal.

## 2. Train a model

```bash
digits-starter train --output-dir runs/first --seed 42
```

The command creates a stratified **60% training / 20% validation / 20% test** split. Counts are rounded to whole images. The scaler and classifier are fitted only on the training split. It then evaluates all three splits and saves:

```text
runs/first/
├── model.joblib           # Fitted preprocessing + classifier + split seed
├── metrics.json           # Accuracy, counts, per-class metrics, confusion matrix
├── confusion_matrix.png  # Rows: true label; columns: predicted label
├── predictions.png       # First test images with labels and predictions
├── mistakes.png          # Up to 16 incorrect test predictions
└── predictions.csv       # Dataset index, true label, predicted label, correctness
```

Use a new output directory for each experiment. Existing nonempty directories and model files are protected from replacement. Model weights and generated results are excluded from Git.

## 3. Predict an unseen dataset image

```bash
digits-starter predict --model runs/first/model.joblib --index 0
```

The default source is the test split. `--index` selects a position **within that split**, not a global dataset index. The JSON output includes the true label, predicted label, original dataset index, and probabilities for all ten classes.

To choose another split explicitly:

```bash
digits-starter predict --model runs/first/model.joblib --part validation --index 5
```

## 4. Evaluate the saved model

```bash
digits-starter evaluate \
  --model runs/first/model.joblib \
  --output-dir runs/evaluation
```

This loads the saved pipeline and recreates the split from its saved seed. It writes evaluation reports without fitting the model again. Keep the same dependency versions when comparing saved-model results; the versions are stored in the model artifact.

## 5. Try a PNG or JPEG

First export a dataset image to understand the expected appearance:

```bash
digits-starter export --part test --index 0 --seed 42 --output outputs/digit.png
digits-starter predict --model runs/first/model.joblib --image outputs/digit.png
```

For a digit drawn as **dark ink on a white background**, invert its colors:

```bash
digits-starter predict --model runs/first/model.joblib --image my_digit.png --invert
```

The program converts the image to grayscale, resizes it to 8 × 8, and maps intensities to 0–16. It does not crop or center handwriting automatically. A digit filling a similar area to the dataset examples is a better starting point. Low-resolution preprocessing and differences in handwriting can cause incorrect predictions. The returned probabilities are model outputs, not a guarantee of correctness.

## Small experiments

Keep the seed fixed while changing one setting:

```bash
digits-starter train --output-dir runs/c-01 --seed 42 --c 0.1
digits-starter train --output-dir runs/c-10 --seed 42 --c 10
```

`C` is the inverse regularization strength: a smaller value penalizes large weights more strongly. Compare **validation** accuracy when selecting it. Avoid repeatedly selecting settings based on test accuracy; reserve the test split for the final assessment.

Further exercises and the equations are in [the learning guide](docs/learning-guide.md).

## Code reading order

1. [`data.py`](src/digits_starter/data.py): load images and labels; create separate splits.
2. [`model.py`](src/digits_starter/model.py): preprocessing and the classifier.
3. [`cli.py`](src/digits_starter/cli.py): find `model.fit(x_train, y_train)` and follow the workflow.
4. [`reporting.py`](src/digits_starter/reporting.py): compare predictions against the correct labels.
5. [`artifacts.py`](src/digits_starter/artifacts.py): save and restore the complete model pipeline.
6. [`images.py`](src/digits_starter/images.py): preprocess an optional external image.

## Tests you can run later

```bash
python -m pytest -q
```

The pytest files cover separate and reproducible data splits, training-only scaler statistics, valid prediction probabilities, saved-model round trips, image preprocessing, output protection, and an end-to-end CLI training/prediction workflow. These tests are included but **have not been executed** during repository creation.

GitHub Actions is configured for **manual execution only**: open **Actions → Manual tests → Run workflow** if you want to run it later. There are no automatic test or training runs on push.

## Notes

- `python -m digits_starter` is an alternative to the `digits-starter` command after installation.
- Use `digits-starter train --help` to see the available options.
- Load only your own generated `.joblib` models: joblib uses Python's pickle mechanism internally.
- No dataset files, personal images, trained weights, virtual environments, or generated reports are committed.

