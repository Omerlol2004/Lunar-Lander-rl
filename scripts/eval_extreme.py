#!/usr/bin/env python3
"""
Extreme Condition Evaluation Script.
Evaluates model under high wind (25.0) and high turbulence (2.5) conditions.
"""

import os
import json
import argparse
from lunar_lander.policy import PolicyEngine
from lunar_lander.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model in extreme conditions")
    parser.add_argument("--model-path", type=str, default="models/best_model.zip", help="Path to model file")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--vecnorm-path", type=str, default="models/lander_vecnorm.pkl", help="Path to VecNormalize stats")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file {args.model_path} not found.")
        return

    print("--- Running EXTREME Disturbance Evaluation ---")
    engine = PolicyEngine(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path if os.path.exists(args.vecnorm_path or "") else None,
        device="cpu",
    )
    evaluator = Evaluator(engine)
    metrics = evaluator.evaluate(regime="extreme", episodes=args.episodes, seed=args.seed)

    results = {
        "extreme_mean": metrics.mean_reward,
        "extreme_std": metrics.std_reward,
        "median": metrics.median_reward,
        "iqr": metrics.iqr_reward,
        "min": metrics.min_reward,
        "max": metrics.max_reward,
        "success_rate_percent": metrics.success_rate_percent,
        "episodes": metrics.episodes,
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()