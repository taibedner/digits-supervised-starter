import json
import subprocess
import sys

import pytest


def command(*args):
    return subprocess.run([sys.executable, "-m", "digits_starter", *map(str, args)],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("name", ["train", "evaluate", "predict", "preview", "export"])
def test_command_help(name):
    result = command(name, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_complete_training_and_prediction_workflow(tmp_path):
    run = tmp_path / "run"
    trained = command("train", "--output-dir", run)
    assert trained.returncode == 0, trained.stderr
    assert (run / "model.joblib").is_file()
    assert (run / "metrics.json").is_file()
    assert (run / "confusion_matrix.png").is_file()
    assert (run / "predictions.png").is_file()
    assert (run / "mistakes.png").is_file()
    assert (run / "predictions.csv").is_file()
    result = command("predict", "--model", run / "model.joblib", "--index", "0")
    assert result.returncode == 0, result.stderr
    prediction = json.loads(result.stdout)
    assert prediction["predicted_label"] in range(10)
    assert prediction["true_label"] in range(10)
    assert len(prediction["probabilities"]) == 10
    evaluated = command("evaluate", "--model", run / "model.joblib", "--output-dir", tmp_path / "evaluation")
    assert evaluated.returncode == 0, evaluated.stderr
    assert json.loads(trained.stdout)["accuracy"] == json.loads(evaluated.stdout)["accuracy"]


def test_invalid_training_parameter_creates_no_artifacts(tmp_path):
    output = tmp_path / "bad"
    result = command("train", "--c", "-1", "--output-dir", output)
    assert result.returncode == 2
    assert not output.exists()

