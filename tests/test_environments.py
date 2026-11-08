"""
Unit tests for EnvironmentFactory and simulation regimes.
"""

import pytest
import gymnasium as gym
import numpy as np
from lunar_lander.envs import EnvironmentFactory


def test_create_clean_environment():
    vec_env = EnvironmentFactory.create(regime="clean", seed=123)
    obs = vec_env.reset()

    assert obs.shape == (1, 8)
    assert vec_env.action_space.n == 4

    action = np.array([0])
    next_obs, reward, done, info = vec_env.step(action)
    assert next_obs.shape == (1, 8)
    assert isinstance(float(reward[0]), float)
    assert isinstance(bool(done[0]), bool)

    vec_env.close()


def test_create_robust_environment():
    vec_env = EnvironmentFactory.create(regime="robust", seed=456)
    obs = vec_env.reset()
    assert obs.shape == (1, 8)
    vec_env.close()


def test_create_extreme_environment():
    vec_env = EnvironmentFactory.create(regime="extreme", seed=789)
    obs = vec_env.reset()
    assert obs.shape == (1, 8)
    vec_env.close()


def test_invalid_regime_raises():
    with pytest.raises(ValueError):
        EnvironmentFactory.create(regime="invalid_regime")
