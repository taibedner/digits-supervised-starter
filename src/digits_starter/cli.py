"""Entry points for preview -> train -> predict -> evaluate."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .artifacts import load_model, new_output_dir, save_model
from .data import load_split
from .images import export_digit, image_features
from .model import make_model
from .reporting import sample_grid, save_report


def train(args):
    split = load_split(args.seed)
    model = make_model(c=args.c, max_iter=args.max_iter, seed=args.seed)
    directory = new_output_dir(args.output_dir)
    x_train, y_train = split.arrays("train")
    # Supervised learning: each input image is paired with its correct digit label.
    model.fit(x_train, y_train)
    save_model(directory / "model.joblib", model, seed=args.seed, c=args.c, max_iter=args.max_iter)
    scores = save_report(directory, model, split, {
        "dataset": "sklearn.load_digits", "seed": args.seed,
        "parameters": {"C": args.c, "max_iter": args.max_iter},
    })
    return {"output_dir": str(directory), "accuracy": {part: row["accuracy"] for part, row in scores.items()}}


def evaluate(args):
    payload = load_model(args.model)
    split = load_split(payload["seed"])
    directory = new_output_dir(args.output_dir)
    scores = save_report(directory, payload["model"], split, {
        "model": str(args.model), "seed": payload["seed"], "parameters": payload["parameters"],
    })
    return {"output_dir": str(directory), "accuracy": {part: row["accuracy"] for part, row in scores.items()}}


def _sample(split, part, index):
    indices = split.indices(part)
    if not 0 <= index < len(indices):
        raise ValueError(f"index for {part} must be between 0 and {len(indices) - 1}")
    dataset_index = int(indices[index])
    return dataset_index, split.images[dataset_index], int(split.targets[dataset_index])


def predict(args):
    payload = load_model(args.model)
    if args.image is not None:
        features = image_features(args.image, invert=args.invert)
        result = {"source": str(args.image)}
    else:
        if args.invert:
            raise ValueError("--invert applies only to --image")
        split = load_split(payload["seed"])
        index, image, label = _sample(split, args.part, args.index if args.index is not None else 0)
        features = image.reshape(1, 64)
        result = {"split": args.part, "dataset_index": index, "true_label": label}
    model = payload["model"]
    predicted = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    return {**result, "predicted_label": predicted,
            "probabilities": {str(int(label)): float(value) for label, value in zip(model.classes_, probabilities)}}


def preview(args):
    split = load_split(args.seed)
    indices = split.indices(args.part)[:16]
    sample_grid(split.images[indices], split.targets[indices], args.output)
    return {"output": str(args.output), "split": args.part, "samples": len(indices)}


def export(args):
    split = load_split(args.seed)
    index, image, label = _sample(split, args.part, args.index)
    export_digit(image, args.output)
    return {"output": str(args.output), "true_label": label, "dataset_index": index}


def parser():
    root = argparse.ArgumentParser(prog="digits-starter", description="Learn supervised image classification with handwritten digits")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)

    command = sub.add_parser("train", help="fit on training data, then save the model and evaluation")
    command.add_argument("--output-dir", type=Path, default=Path("runs/first"))
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--c", type=float, default=1.0, help="inverse regularization strength")
    command.add_argument("--max-iter", type=int, default=2000)
    command.set_defaults(handler=train)

    command = sub.add_parser("evaluate", help="evaluate a saved model on the same split without retraining")
    command.add_argument("--model", type=Path, default=Path("runs/first/model.joblib"))
    command.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    command.set_defaults(handler=evaluate)

    command = sub.add_parser("predict", help="classify one image using a saved model")
    command.add_argument("--model", type=Path, default=Path("runs/first/model.joblib"))
    source = command.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, help="position in the selected split (default: 0)")
    source.add_argument("--image", type=Path, help="optional PNG/JPEG file")
    command.add_argument("--part", choices=("train", "validation", "test"), default="test")
    command.add_argument("--invert", action="store_true", help="invert a white-background image")
    command.set_defaults(handler=predict)

    for name, handler, help_text in [("preview", preview, "save a grid of labeled digit images"),
                                      ("export", export, "export one dataset image as a PNG")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--part", choices=("train", "validation", "test"), default="train" if name == "preview" else "test")
        command.add_argument("--output", type=Path, default=Path(f"outputs/{name}.png"))
        if name == "export":
            command.add_argument("--index", type=int, default=0)
        command.set_defaults(handler=handler)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
