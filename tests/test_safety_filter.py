"""
Unit tests for Defensive Safety Filter and Emergency Attitude Recovery.
"""

import numpy as np
import pytest
from lunar_lander.safety import SafetyFilter, SafetyDecision
from lunar_lander.config import SafetyConfig


@pytest.fixture
def safety_filter():
    config = SafetyConfig(
        max_tilt_rad=0.45,
        max_angular_velocity=2.5,
        max_descent_velocity_near_ground=1.2,
        ground_proximity_threshold=0.25,
        enable_failsafe_override=True,
    )
    return SafetyFilter(config)


def test_nominal_state_no_override(safety_filter):
    # Nominal upright state hovering smoothly
    obs = np.array([0.0, 0.8, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    decision = safety_filter.evaluate(obs, policy_action=0)

    assert not decision.overridden
    assert decision.action == 0
    assert decision.original_action == 0
    assert decision.reason == "POLICY_NOMINAL"
    assert not decision.is_ood


def test_emergency_de_tumble_counter_clockwise(safety_filter):
    # Excessive counter-clockwise angular velocity (spinning violently)
    obs = np.array([0.0, 0.5, 0.0, -0.2, 0.1, 3.2, 0.0, 0.0], dtype=np.float32)
    decision = safety_filter.evaluate(obs, policy_action=0)

    assert decision.overridden
    assert decision.action == 3  # Fire right engine to induce CW torque
    assert decision.original_action == 0
    assert "EMERGENCY_DE_TUMBLE_CCW" in decision.reason


def test_emergency_de_tumble_clockwise(safety_filter):
    # Excessive clockwise angular velocity
    obs = np.array([0.0, 0.5, 0.0, -0.2, -0.1, -3.2, 0.0, 0.0], dtype=np.float32)
    decision = safety_filter.evaluate(obs, policy_action=0)

    assert decision.overridden
    assert decision.action == 1  # Fire left engine to induce CCW torque
    assert decision.original_action == 0
    assert "EMERGENCY_DE_TUMBLE_CW" in decision.reason


def test_critical_tilt_attitude_recovery_left(safety_filter):
    # Lander tilted too far left (angle > max_tilt_rad)
    obs = np.array([0.0, 0.6, 0.0, -0.2, 0.55, 0.1, 0.0, 0.0], dtype=np.float32)
    decision = safety_filter.evaluate(obs, policy_action=0)

    assert decision.overridden
    assert decision.action == 3  # Fire right engine
    assert "ATTITUDE_RECOVERY_TILT_LEFT" in decision.reason


def test_critical_tilt_attitude_recovery_right(safety_filter):
    # Lander tilted too far right (angle < -max_tilt_rad)
    obs = np.array([0.0, 0.6, 0.0, -0.2, -0.55, -0.1, 0.0, 0.0], dtype=np.float32)
    decision = safety_filter.evaluate(obs, policy_action=0)

    assert decision.overridden
    assert decision.action == 1  # Fire left engine
    assert "ATTITUDE_RECOVERY_TILT_RIGHT" in decision.reason


def test_touchdown_descent_brake(safety_filter):
    # Low altitude (y=0.15 < 0.25) descending fast (vy=-1.8 < -1.2)
    obs = np.array([0.0, 0.15, 0.0, -1.8, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)
    decision = safety_filter.evaluate(obs, policy_action=0)

    assert decision.overridden
    assert decision.action == 2  # Fire main thruster to cushion landing
    assert "TOUCHDOWN_DESCENT_BRAKE" in decision.reason


def test_out_of_distribution_detection(safety_filter):
    # State with extreme horizontal position and velocities
    obs = np.array([3.5, 1.0, 6.2, -7.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    is_ood, anomalies = safety_filter.inspect_observation(obs)

    assert is_ood
    assert len(anomalies) >= 3

    decision = safety_filter.evaluate(obs, policy_action=0)
    assert decision.is_ood


def test_safety_override_disabled():
    disabled_filter = SafetyFilter(SafetyConfig(enable_failsafe_override=False))
    # Extreme tilt, but override is disabled
    obs = np.array([0.0, 0.6, 0.0, -0.2, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    decision = disabled_filter.evaluate(obs, policy_action=0)

    assert not decision.overridden
    assert decision.action == 0
    assert decision.reason == "DEFENSIVE_OVERRIDE_DISABLED"
