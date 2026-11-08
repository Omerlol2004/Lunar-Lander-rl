"""
Unit tests for PolicyEngine (inference, normalization, batching, confidence).
"""

import os
import numpy as np
import pytest
from lunar_lander.policy import PolicyEngine, ACTION_NAMES


@pytest.fixture(scope="session")
def engine():
    model_path = "models/best_model.zip"
    vecnorm_path = "models/lander_vecnorm.pkl"
    if not os.path.exists(model_path):
        pytest.skip("Model checkpoint not found")
    return PolicyEngine(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        device="cpu",
        warmup=True,
    )


def test_predict_single_nominal(engine):
    obs = np.array([0.0, 0.7, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = engine.predict(obs, deterministic=True, apply_safety=True)

    assert "action" in result
    assert result["action"] in [0, 1, 2, 3]
    assert result["action_name"] in ACTION_NAMES.values()
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["critic_value"], float)
    assert "latency_ms" in result
    assert result["latency_ms"] < 25.0  # Ultra-fast CPU inference


def test_predict_deterministic_consistency(engine):
    obs = np.array([0.15, 0.45, -0.05, -0.25, 0.03, -0.01, 0.0, 0.0], dtype=np.float32)
    res1 = engine.predict(obs, deterministic=True)
    res2 = engine.predict(obs, deterministic=True)

    assert res1["action"] == res2["action"]
    assert res1["raw_policy_action"] == res2["raw_policy_action"]
    assert np.isclose(res1["confidence"], res2["confidence"], atol=1e-5)
    assert np.isclose(res1["critic_value"], res2["critic_value"], atol=1e-4)


def test_predict_probabilities_sum_to_one(engine):
    obs = np.array([0.0, 0.5, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = engine.predict(obs)

    probs = list(result["probabilities"].values())
    assert len(probs) == 4
    assert np.isclose(sum(probs), 1.0, atol=1e-5)


def test_predict_invalid_shape_raises(engine):
    with pytest.raises(ValueError):
        engine.predict(np.zeros(7, dtype=np.float32))

    with pytest.raises(ValueError):
        engine.predict(np.zeros((2, 8), dtype=np.float32))


def test_predict_batch_matches_single(engine):
    obs1 = np.array([0.0, 0.8, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs2 = np.array([0.2, 0.4, 0.1, -0.3, 0.05, 0.02, 0.0, 0.0], dtype=np.float32)
    obs3 = np.array([-0.2, 0.3, -0.1, -0.4, -0.05, -0.02, 0.0, 0.0], dtype=np.float32)

    batch_obs = np.stack([obs1, obs2, obs3])
    batch_results = engine.predict_batch(batch_obs, deterministic=True, apply_safety=True)

    assert len(batch_results) == 3

    for i, single_obs in enumerate([obs1, obs2, obs3]):
        single_res = engine.predict(single_obs, deterministic=True, apply_safety=True)
        assert batch_results[i]["action"] == single_res["action"]
        assert batch_results[i]["raw_policy_action"] == single_res["raw_policy_action"]
        assert np.isclose(batch_results[i]["confidence"], single_res["confidence"], atol=1e-4)
