import numpy as np
import pytest

from digits_starter.artifacts import load_model, new_output_dir, save_model
from digits_starter.model import make_model
from digits_starter.reporting import evaluate_model


def test_scaler_learns_training_statistics_only(fitted_model, split):
    x, _ = split.arrays("train")
    np.testing.assert_allclose(fitted_model.named_steps["scaler"].mean_, x.mean(axis=0))
    assert fitted_model.named_steps["scaler"].n_samples_seen_ == len(x)


def test_predictions_are_valid_probabilities(fitted_model, split):
    x, _ = split.arrays("test")
    probabilities = fitted_model.predict_proba(x)
    assert probabilities.shape == (len(x), 10)
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert set(fitted_model.predict(x)) <= set(range(10))


def test_saved_model_preserves_predictions(fitted_model, split, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(path, fitted_model, seed=42, c=1.0, max_iter=2000)
    restored = load_model(path)
    x, _ = split.arrays("test")
    np.testing.assert_allclose(restored["model"].predict_proba(x), fitted_model.predict_proba(x))
    assert restored["seed"] == 42
    with pytest.raises(FileExistsError):
        save_model(path, fitted_model, seed=42, c=1.0, max_iter=2000)


def test_evaluation_counts_match_test_samples(fitted_model, split):
    scores = evaluate_model(fitted_model, split)
    matrix = np.asarray(scores["test"]["confusion_matrix"])
    assert matrix.shape == (10, 10)
    assert matrix.sum() == len(split.test_indices)
    for part in ("train", "validation", "test"):
        assert 0 <= scores[part]["accuracy"] <= 1


@pytest.mark.parametrize("c", [0, -1, float("nan"), float("inf")])
def test_invalid_regularization(c):
    with pytest.raises(ValueError):
        make_model(c=c)


def test_output_directory_does_not_replace_prior_work(tmp_path):
    directory = new_output_dir(tmp_path / "run")
    (directory / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        new_output_dir(directory)
    assert (directory / "keep.txt").read_text() == "keep"

