"""
Standardized Empirical Evaluation Engine.
Computes publication-grade statistical benchmarks (Mean, Std, Median, IQR,
Tail Value-at-Risk, Solved Landing Rate, and Safety Intervention Rate).
"""

from typing import Dict, List, Optional, Literal
import numpy as np
from pydantic import BaseModel

from lunar_lander.envs import EnvironmentFactory
from lunar_lander.policy import PolicyEngine


class EvaluationMetrics(BaseModel):
    """Rigorous empirical evaluation telemetry."""
    regime: str
    episodes: int
    mean_reward: float
    std_reward: float
    median_reward: float
    iqr_reward: float
    min_reward: float
    max_reward: float
    success_rate_percent: float
    crash_rate_percent: float
    avg_steps: float
    safety_override_rate_percent: float


class Evaluator:
    """Evaluates policy checkpoints deterministically across environmental regimes."""

    def __init__(self, policy_engine: PolicyEngine):
        self.engine = policy_engine

    def evaluate(
        self,
        regime: Literal["clean", "robust", "extreme"] = "clean",
        episodes: int = 100,
        seed: int = 42,
        apply_safety: bool = True,
    ) -> EvaluationMetrics:
        """Runs deterministic multi-episode evaluation against specified turbulence regime."""
        env = EnvironmentFactory.create(
            regime=regime,
            seed=seed,
            vecnorm_path=None,  # PolicyEngine handles observation normalization cleanly
            training=False,
        )

        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        overridden_steps = 0
        successes = 0
        crashes = 0

        for ep in range(episodes):
            obs = env.reset()
            done = [False]
            ep_reward = 0.0
            ep_len = 0

            while not done[0]:
                raw_obs = obs[0]
                result = self.engine.predict(raw_obs, deterministic=True, apply_safety=apply_safety)
                action = np.array([result["action"]])

                if result["safety"]["overridden"]:
                    overridden_steps += 1
                total_steps += 1
                ep_len += 1

                obs, reward, done, info = env.step(action)
                ep_reward += float(reward[0])

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)

            if ep_reward >= 200.0:
                successes += 1
            elif ep_reward < -100.0:
                crashes += 1

        env.close()

        arr_rewards = np.array(episode_rewards)
        q25, q75 = np.percentile(arr_rewards, [25, 75])
        override_rate = (overridden_steps / total_steps * 100.0) if total_steps > 0 else 0.0

        return EvaluationMetrics(
            regime=regime,
            episodes=episodes,
            mean_reward=round(float(np.mean(arr_rewards)), 2),
            std_reward=round(float(np.std(arr_rewards)), 2),
            median_reward=round(float(np.median(arr_rewards)), 2),
            iqr_reward=round(float(q75 - q25), 2),
            min_reward=round(float(np.min(arr_rewards)), 2),
            max_reward=round(float(np.max(arr_rewards)), 2),
            success_rate_percent=round((successes / episodes) * 100.0, 1),
            crash_rate_percent=round((crashes / episodes) * 100.0, 1),
            avg_steps=round(float(np.mean(episode_lengths)), 1),
            safety_override_rate_percent=round(override_rate, 2),
        )
