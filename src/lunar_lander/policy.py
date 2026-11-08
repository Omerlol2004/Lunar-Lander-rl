"""
Thread-safe, low-latency Policy Inference Engine.
Wraps the trained PPO ActorCriticPolicy with preserved VecNormalize statistics,
confidence gating, critic value estimation, and safety filtering.
"""

import time
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from lunar_lander.safety import SafetyFilter, SafetyDecision
from lunar_lander.config import SafetyConfig

ACTION_NAMES = {
    0: "COAST",
    1: "FIRE_LEFT_ORIENTATION_ENGINE",
    2: "FIRE_MAIN_THRUSTER",
    3: "FIRE_RIGHT_ORIENTATION_ENGINE",
}


class PolicyEngine:
    """
    Production-grade inference engine providing deterministic decision making,
    confidence metrics, state value estimation, and safety overrides.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.zip",
        vecnorm_path: Optional[str] = "models/lander_vecnorm.pkl",
        device: str = "cpu",
        safety_config: Optional[SafetyConfig] = None,
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.device = torch.device(device)
        self.safety_filter = SafetyFilter(safety_config or SafetyConfig())

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # Load VecNormalize statistics if provided
        self.vec_env: Optional[VecNormalize] = None
        if vecnorm_path and os.path.exists(vecnorm_path):
            dummy_env = DummyVecEnv([lambda: gym.make("LunarLander-v3")])
            self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            self.vec_env.norm_obs = True

        # Load PPO model
        self.model = PPO.load(model_path, device=self.device)
        self.model.policy.eval()

        # Cache running observation normalization statistics locally for thread-safety & speed
        if self.vec_env is not None and hasattr(self.vec_env, "obs_rms"):
            self.obs_mean = np.array(self.vec_env.obs_rms.mean, dtype=np.float32)
            self.obs_var = np.array(self.vec_env.obs_rms.var, dtype=np.float32)
            self.obs_epsilon = float(self.vec_env.epsilon)
            self.clip_obs = float(self.vec_env.clip_obs)
        else:
            self.obs_mean = None
            self.obs_var = None

        if warmup:
            self._warmup()

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply deterministic running observation normalization without mutating state."""
        if self.obs_mean is not None and self.obs_var is not None:
            norm_obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.obs_epsilon)
            return np.clip(norm_obs, -self.clip_obs, self.clip_obs).astype(np.float32)
        return obs.astype(np.float32)

    def _warmup(self):
        """Warm up PyTorch JIT and inference pathways."""
        dummy_state = np.zeros(8, dtype=np.float32)
        self.predict(dummy_state, deterministic=True)

    def predict(
        self,
        observation: Union[List[float], np.ndarray],
        deterministic: bool = True,
        apply_safety: bool = True,
    ) -> Dict:
        """
        Executes policy forward pass on a single telemetry state vector.

        Returns structured inference telemetry including action, probabilities,
        critic value estimate, safety override status, and latency.
        """
        t0 = time.perf_counter()
        raw_obs = np.asarray(observation, dtype=np.float32)
        if raw_obs.shape != (8,):
            raise ValueError(f"Observation must be an 8-dimensional vector, got shape {raw_obs.shape}")

        norm_obs = self._normalize_observation(raw_obs)
        th_obs = torch.as_tensor(norm_obs, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.model.policy.get_distribution(th_obs)
            probs = dist.distribution.probs.cpu().numpy()[0]
            val = float(self.model.policy.predict_values(th_obs).cpu().numpy()[0][0])
            if deterministic:
                policy_act = int(np.argmax(probs))
            else:
                policy_act = int(dist.get_actions().cpu().numpy()[0])

        # Apply defensive safety filter
        if apply_safety:
            safety_decision = self.safety_filter.evaluate(raw_obs, policy_act)
            final_action = safety_decision.action
        else:
            safety_decision = SafetyDecision(
                action=policy_act,
                original_action=policy_act,
                overridden=False,
                reason="SAFETY_DISABLED",
                is_ood=False,
            )
            final_action = policy_act

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "action": final_action,
            "action_name": ACTION_NAMES.get(final_action, "UNKNOWN"),
            "raw_policy_action": policy_act,
            "probabilities": {ACTION_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "confidence": float(probs[final_action]),
            "critic_value": val,
            "safety": safety_decision.model_dump(),
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(
        self,
        observations: Union[List[List[float]], np.ndarray],
        deterministic: bool = True,
        apply_safety: bool = True,
    ) -> List[Dict]:
        """Vectorized batch inference across multiple state observations."""
        t0 = time.perf_counter()
        raw_batch = np.asarray(observations, dtype=np.float32)
        if raw_batch.ndim != 2 or raw_batch.shape[1] != 8:
            raise ValueError(f"Expected batch shape (N, 8), got {raw_batch.shape}")

        norm_batch = self._normalize_observation(raw_batch)
        th_batch = torch.as_tensor(norm_batch, device=self.device)

        with torch.no_grad():
            dist = self.model.policy.get_distribution(th_batch)
            probs_matrix = dist.distribution.probs.cpu().numpy()
            vals_matrix = self.model.policy.predict_values(th_batch).cpu().numpy().flatten()
            if deterministic:
                policy_acts = np.argmax(probs_matrix, axis=1)
            else:
                policy_acts = dist.get_actions().cpu().numpy()

        results = []
        for i in range(len(raw_batch)):
            raw_obs = raw_batch[i]
            policy_act = int(policy_acts[i])
            probs = probs_matrix[i]
            val = float(vals_matrix[i])

            if apply_safety:
                decision = self.safety_filter.evaluate(raw_obs, policy_act)
                final_action = decision.action
            else:
                decision = SafetyDecision(
                    action=policy_act,
                    original_action=policy_act,
                    overridden=False,
                    reason="SAFETY_DISABLED",
                    is_ood=False,
                )
                final_action = policy_act

            results.append({
                "action": final_action,
                "action_name": ACTION_NAMES.get(final_action, "UNKNOWN"),
                "raw_policy_action": policy_act,
                "probabilities": {ACTION_NAMES[j]: float(p) for j, p in enumerate(probs)},
                "confidence": float(probs[final_action]),
                "critic_value": val,
                "safety": decision.model_dump(),
            })

        return results
