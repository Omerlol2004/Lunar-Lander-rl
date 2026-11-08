#!/usr/bin/env python3
"""
Playback script for Lunar Lander demonstration and video recording.
Generates demonstration recordings for Clean and Robust regimes.
"""

import os
import argparse
import numpy as np
import imageio
from lunar_lander.policy import PolicyEngine
from lunar_lander.envs import EnvironmentFactory


def record_regime_video(
    engine: PolicyEngine,
    regime: str,
    output_path: str,
    episodes: int = 1,
    seed: int = 42,
    fps: int = 30,
):
    """Record an episode in the specified regime and export as a GIF."""
    print(f"Recording {regime.upper()} demonstration to '{output_path}'...")
    env = EnvironmentFactory.create(
        regime=regime,
        seed=seed,
        render_mode="rgb_array",
        vecnorm_path=engine.vecnorm_path,
        training=False,
    )

    frames = []
    total_reward = 0.0

    for ep in range(episodes):
        obs = env.reset()
        done = [False]
        ep_reward = 0.0

        while not done[0]:
            frame = env.render()
            if frame is not None and frame.size > 0:
                frames.append(frame)

            raw_obs = obs[0]
            result = engine.predict(raw_obs, deterministic=True, apply_safety=True)
            action = np.array([result["action"]])

            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])

        total_reward += ep_reward
        print(f"Episode {ep + 1}/{episodes} finished with reward: {ep_reward:.2f}")

    env.close()

    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path} ({len(frames)} frames)")
    else:
        print("Warning: No video frames captured.")


def main():
    parser = argparse.ArgumentParser(description="Generate demonstration videos of Lunar Lander policy")
    parser.add_argument("--model", type=str, default="models/best_model.zip", help="Path to model file")
    parser.add_argument("--vecnorm", type=str, default="models/lander_vecnorm.pkl", help="Path to VecNormalize file")
    parser.add_argument("--output-dir", type=str, default="videos", help="Output directory")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per regime")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    engine = PolicyEngine(model_path=args.model, vecnorm_path=args.vecnorm, device="cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    record_regime_video(
        engine,
        regime="clean",
        output_path=os.path.join(args.output_dir, "lander_clean.gif"),
        episodes=args.episodes,
        fps=args.fps,
    )

    record_regime_video(
        engine,
        regime="robust",
        output_path=os.path.join(args.output_dir, "lander_robust.gif"),
        episodes=args.episodes,
        fps=args.fps,
    )

    print("\nPlayback demonstration generation complete.")


if __name__ == "__main__":
    main()