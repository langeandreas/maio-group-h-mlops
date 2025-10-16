"""Tests that the training script produces expected artifacts and metrics."""

import json
import pathlib
import subprocess
import sys

def test_training_artifacts_exist():
    """Test that the training script produces expected artifacts and metrics."""
    subprocess.check_call([sys.executable, "modelling.py"])
    assert pathlib.Path("models/model.pkl").exists()
    m = json.loads(pathlib.Path("models/training_metrics.txt").read_text(encoding="utf-8"))
    assert "rmse" in m
