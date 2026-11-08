"""
Defensive Safety Filter and Fail-Safe Attitude / Touchdown Override Engine.

Provides deterministic safety gating for neural policy actions in autonomous
descent control, mitigating catastrophic failure under extreme aerodynamic shock.
"""

from typing import Tuple, List, Union
import numpy as np
from pydantic import BaseModel
from lunar_lander.config import SafetyConfig


class SafetyDecision(BaseModel):
    """Result of safety filtering on an action decision."""
    action: int
    original_action: int
    overridden: bool
    reason: str
    is_ood: bool


class SafetyFilter:
    """
    Defensive runtime guardrail monitoring telemetry states and enforcing
    aerospace flight envelope constraints.
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()

    def inspect_observation(self, obs: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check if observation vector is within physical operational limits.
        Observation layout: [x, y, vx, vy, angle, angular_velocity, leg1, leg2]
        """
        anomalies = []
        is_ood = False

        if len(obs) != 8:
            return True, [f"Invalid observation dimension: expected 8, got {len(obs)}"]

        x, y, vx, vy, angle, ang_vel, l1, l2 = obs

        # Out of distribution checks
        if abs(x) > 2.0:
            anomalies.append(f"Excessive horizontal deviation: x={x:.2f}")
            is_ood = True
        if y < -0.1 or y > 2.0:
            anomalies.append(f"Altitude anomaly: y={y:.2f}")
            is_ood = True
        if abs(vx) > 5.0:
            anomalies.append(f"Excessive horizontal velocity: vx={vx:.2f}")
            is_ood = True
        if abs(vy) > 5.0:
            anomalies.append(f"Excessive vertical velocity: vy={vy:.2f}")
            is_ood = True
        if abs(angle) > np.pi:
            anomalies.append(f"Lander inverted: angle={angle:.2f} rad")
            is_ood = True

        return is_ood, anomalies

    def evaluate(self, obs: np.ndarray, policy_action: int) -> SafetyDecision:
        """
        Evaluates the telemetry observation against safety boundaries.
        Intervenes deterministically if the policy action risks vehicle loss.

        Action mapping:
          0: Coast (no thrust)
          1: Fire left engine (induces rightward torque / clockwise)
          2: Fire main engine (vertical deceleration)
          3: Fire right engine (induces leftward torque / counterclockwise)
        """
        obs = np.asarray(obs, dtype=np.float32)
        is_ood, anomalies = self.inspect_observation(obs)

        if not self.config.enable_failsafe_override or len(obs) != 8:
            return SafetyDecision(
                action=int(policy_action),
                original_action=int(policy_action),
                overridden=False,
                reason="DEFENSIVE_OVERRIDE_DISABLED" if not self.config.enable_failsafe_override else "INVALID_STATE",
                is_ood=is_ood
            )

        x, y, vx, vy, angle, ang_vel, l1, l2 = obs

        # Rule 1: High angular velocity tumbling mitigation
        if ang_vel > self.config.max_angular_velocity:
            # Spinning counter-clockwise rapidly -> Fire right engine to counteract
            if policy_action != 3:
                return SafetyDecision(
                    action=3,
                    original_action=int(policy_action),
                    overridden=True,
                    reason=f"EMERGENCY_DE_TUMBLE_CCW (ang_vel={ang_vel:.2f} > {self.config.max_angular_velocity})",
                    is_ood=is_ood
                )
        elif ang_vel < -self.config.max_angular_velocity:
            # Spinning clockwise rapidly -> Fire left engine to counteract
            if policy_action != 1:
                return SafetyDecision(
                    action=1,
                    original_action=int(policy_action),
                    overridden=True,
                    reason=f"EMERGENCY_DE_TUMBLE_CW (ang_vel={ang_vel:.2f} < -{self.config.max_angular_velocity})",
                    is_ood=is_ood
                )

        # Rule 2: Critical attitude tilt threshold
        if angle > self.config.max_tilt_rad:
            # Tilted left -> Fire right engine to torque clockwise
            if policy_action != 3:
                return SafetyDecision(
                    action=3,
                    original_action=int(policy_action),
                    overridden=True,
                    reason=f"ATTITUDE_RECOVERY_TILT_LEFT (angle={angle:.2f} rad)",
                    is_ood=is_ood
                )
        elif angle < -self.config.max_tilt_rad:
            # Tilted right -> Fire left engine to torque counterclockwise
            if policy_action != 1:
                return SafetyDecision(
                    action=1,
                    original_action=int(policy_action),
                    overridden=True,
                    reason=f"ATTITUDE_RECOVERY_TILT_RIGHT (angle={angle:.2f} rad)",
                    is_ood=is_ood
                )

        # Rule 3: Low-altitude vertical descent brake
        if y < self.config.ground_proximity_threshold and vy < -self.config.max_descent_velocity_near_ground:
            # Descending too fast close to ground -> Fire main thruster to prevent crash
            if policy_action != 2:
                return SafetyDecision(
                    action=2,
                    original_action=int(policy_action),
                    overridden=True,
                    reason=f"TOUCHDOWN_DESCENT_BRAKE (y={y:.2f}, vy={vy:.2f})",
                    is_ood=is_ood
                )

        # Nominal policy execution
        return SafetyDecision(
            action=int(policy_action),
            original_action=int(policy_action),
            overridden=False,
            reason="POLICY_NOMINAL",
            is_ood=is_ood
        )
