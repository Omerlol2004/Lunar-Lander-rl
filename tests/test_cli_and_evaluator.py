"""
Unit tests for Evaluator and CLI commands.
"""

import os
import pytest
from lunar_lander.policy import PolicyEngine
from lunar_lander.evaluator import Evaluator
from lunar_lander.cli import cmd_benchmark, cmd_verify
from unittest.mock import MagicMock


def test_evaluator_clean_regime_short():
    engine = PolicyEngine("models/best_model.zip", "models/lander_vecnorm.pkl", device="cpu")
    evaluator = Evaluator(engine)

    metrics = evaluator.evaluate(regime="clean", episodes=2, seed=42)
    assert metrics.regime == "clean"
    assert metrics.episodes == 2
    assert isinstance(metrics.mean_reward, float)
    assert isinstance(metrics.median_reward, float)
    assert metrics.success_rate_percent >= 0.0


def test_cli_benchmark_runs():
    mock_args = MagicMock()
    mock_args.model = "models/best_model.zip"
    mock_args.vecnorm = "models/lander_vecnorm.pkl"
    mock_args.device = "cpu"
    mock_args.iterations = 50

    # Must complete without exception
    cmd_benchmark(mock_args)


def test_cli_verify_manifest():
    mock_args = MagicMock()
    mock_args.checksum_file = "CHECKSUMS.txt"

    # cmd_verify calls sys.exit(0) on success
    with pytest.raises(SystemExit) as exc_info:
        cmd_verify(mock_args)
    assert exc_info.value.code == 0
