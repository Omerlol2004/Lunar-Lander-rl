"""
Environment Factory for Gymnasium LunarLander-v3.
Provides standardized creation of Clean, Robust, and Extreme turbulence regimes
with reproducible seed management and VecNormalize lifecycle control.
"""

from typing import Optional, Literal
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os


class EnvironmentFactory:
    """Factory for generating standardized simulation environments."""

    @staticmethod
    def create(
        regime: Literal["clean", "robust", "extreme"] = "clean",
        seed: int = 42,
        render_mode: Optional[str] = None,
        vecnorm_path: Optional[str] = None,
        training: bool = False,
        norm_obs: bool = True,
        norm_reward: bool = False,
    ) -> DummyVecEnv:
        """
        Creates a Gymnasium LunarLander-v3 environment wrapped in DummyVecEnv.

        Regimes:
          - clean: Standard deterministic lunar environment (no aerodynamic turbulence).
          - robust: Moderate atmospheric wind and turbulence (wind_power=15.0, turb=1.5).
          - extreme: Severe atmospheric shear (wind_power=25.0, turb=2.5).
        """
        env_kwargs = {}
        if render_mode:
            env_kwargs["render_mode"] = render_mode

        if regime == "clean":
            env_kwargs["enable_wind"] = False
        elif regime == "robust":
            env_kwargs["enable_wind"] = True
            env_kwargs["wind_power"] = 15.0
            env_kwargs["turbulence_power"] = 1.5
        elif regime == "extreme":
            env_kwargs["enable_wind"] = True
            env_kwargs["wind_power"] = 25.0
            env_kwargs["turbulence_power"] = 2.5
        else:
            raise ValueError(f"Unknown regime '{regime}'. Must be one of ['clean', 'robust', 'extreme'].")

        def _init_env():
            env = gym.make("LunarLander-v3", **env_kwargs)
            env.reset(seed=seed)
            return Monitor(env)

        vec_env = DummyVecEnv([_init_env])

        if vecnorm_path and os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = training
            vec_env.norm_obs = norm_obs
            vec_env.norm_reward = norm_reward
        elif vecnorm_path:
            raise FileNotFoundError(f"VecNormalize stats file not found: {vecnorm_path}")

        return vec_env
