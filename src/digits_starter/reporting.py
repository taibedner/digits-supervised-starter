"""Evaluation numbers plus pictures of what the model got right and wrong."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from .artifacts import save_json


def evaluate_model(model, split):
    scores = {}
    for part in ("train", "validation", "test"):
        x, y = split.arrays(part)
        predictions = model.predict(x)
        scores[part] = {"samples": len(y), "accuracy": float(accuracy_score(y, predictions))}
        if part == "test":
            scores[part]["classification_report"] = classification_report(
                y, predictions, labels=list(range(10)), output_dict=True, zero_division=0,
            )
            scores[part]["confusion_matrix"] = confusion_matrix(
                y, predictions, labels=list(range(10)),
            ).tolist()
    return scores


def sample_grid(images, labels, path, predictions=None, title="Labeled digit images"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    try:
        for index, ax in enumerate(axes.flat):
            ax.axis("off")
            if index < min(16, len(images)):
                ax.imshow(images[index], cmap="gray", vmin=0, vmax=16)
                caption = f"Label: {labels[index]}"
                if predictions is not None:
                    caption += f" / Pred: {predictions[index]}"
                ax.set_title(caption, fontsize=9)
        fig.suptitle(title)
        fig.tight_layout()
        with path.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=140)
    finally:
        plt.close(fig)


def save_report(directory, model, split, metadata):
    directory = Path(directory)
    scores = evaluate_model(model, split)
    save_json(directory / "metrics.json", {**metadata, "scores": scores})
    x_test, y_test = split.arrays("test")
    predictions = model.predict(x_test)

    display = ConfusionMatrixDisplay(np.asarray(scores["test"]["confusion_matrix"]), display_labels=list(range(10)))
    display.plot(cmap="Blues", colorbar=False)
    try:
        display.ax_.set_title("Test split: true label vs predicted label")
        display.figure_.tight_layout()
        with (directory / "confusion_matrix.png").open("xb") as handle:
            display.figure_.savefig(handle, format="png", dpi=140)
    finally:
        plt.close(display.figure_)

    test_images = split.images[split.test_indices]
    sample_grid(test_images, y_test, directory / "predictions.png", predictions)
    mistakes = predictions != y_test
    sample_grid(test_images[mistakes], y_test[mistakes], directory / "mistakes.png",
                predictions[mistakes], title=f"Test mistakes: {int(mistakes.sum())}")
    with (directory / "predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_index", "true_label", "predicted_label", "correct"])
        for index, truth, prediction in zip(split.test_indices, y_test, predictions):
            writer.writerow([int(index), int(truth), int(prediction), bool(truth == prediction)])
    return scores

