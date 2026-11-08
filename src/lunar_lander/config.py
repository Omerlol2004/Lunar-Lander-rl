"""
Configuration models with strict Pydantic v2 validation.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


class SafetyConfig(BaseModel):
    """Safety boundary thresholds for spacecraft descent guidance."""
    max_tilt_rad: float = Field(
        default=0.45,  # ~25.8 degrees
        description="Maximum permissible lander tilt angle in radians before triggering attitude recovery."
    )
    max_angular_velocity: float = Field(
        default=2.5,
        description="Maximum permissible angular velocity (rad/s) to prevent uncontrolled tumbling."
    )
    max_descent_velocity_near_ground: float = Field(
        default=1.2,
        description="Maximum downward velocity (m/s) allowed when altitude is below proximity threshold."
    )
    ground_proximity_threshold: float = Field(
        default=0.25,
        description="Altitude threshold (y-coordinate) below which ground descent velocity limits activate."
    )
    enable_failsafe_override: bool = Field(
        default=True,
        description="Whether the safety filter can intervene and override unsafe neural policy actions."
    )


class EnvironmentConfig(BaseModel):
    """Simulation environment parameters for LunarLander-v3."""
    env_id: str = Field(default="LunarLander-v3", description="Gymnasium environment identifier.")
    enable_wind: bool = Field(default=False, description="Enable simulated atmospheric gusts.")
    wind_power: float = Field(default=15.0, description="Baseline aerodynamic wind power force.")
    turbulence_power: float = Field(default=1.5, description="High-frequency wind turbulence force.")
    seed: int = Field(default=0, description="Deterministic pseudo-random seed.")


class ServingConfig(BaseModel):
    """FastAPI production serving parameters."""
    host: str = Field(default="0.0.0.0", description="Network interface binding.")
    port: int = Field(default=8000, description="HTTP listening port.")
    model_path: str = Field(default="models/best_model.zip", description="Path to trained PPO model.")
    vecnorm_path: str = Field(default="models/lander_vecnorm.pkl", description="Path to VecNormalize statistics.")
    enable_safety_filter: bool = Field(default=True, description="Enable defensive safety guardrails.")


class EvaluationConfig(BaseModel):
    """Empirical evaluation protocol parameters."""
    regime: Literal["clean", "robust", "extreme"] = Field(
        default="clean",
        description="Turbulence regime under evaluation."
    )
    episodes: int = Field(default=100, ge=1, le=1000, description="Number of evaluation episodes.")
    seed: int = Field(default=42, description="Root deterministic random seed.")
    deterministic: bool = Field(default=True, description="Deterministic argmax policy execution.")
    model_path: str = Field(default="models/best_model.zip", description="Path to model checkpoint.")
    vecnorm_path: str = Field(default="models/lander_vecnorm.pkl", description="Path to normalization stats.")
