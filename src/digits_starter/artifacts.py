"""Persist a model together with the split seed and its package versions."""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

FORMAT_VERSION = 1


def new_output_dir(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(path, model, *, seed, c, max_iter):
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": "sklearn.load_digits",
        "split": "stratified 60/20/20",
        "seed": seed,
        "parameters": {"C": c, "max_iter": max_iter},
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "model": model,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create keeps an earlier checkpoint intact.
    with path.open("xb") as handle:
        joblib.dump(payload, handle)


def load_model(path):
    # joblib uses pickle internally: load only your own generated model files.
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported model artifact")
    if payload.get("dataset") != "sklearn.load_digits" or "model" not in payload or "seed" not in payload:
        raise ValueError("model artifact is missing its dataset, model or split seed")
    return payload


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

