#!/usr/bin/env python3
"""
SB3 Reference Evaluation Script (Clean and Robust regimes).
Updated to use standardized Gymnasium v1.0+ and the modern LunarLander engine.
"""

import os
import json
import argparse
from lunar_lander.policy import PolicyEngine
from lunar_lander.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model using SB3 reference protocol")
    parser.add_argument("--model-path", type=str, default="models/best_model.zip", help="Path to model checkpoint")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--robust", action="store_true", help="Use robust evaluation with wind and turbulence")
    parser.add_argument("--vecnorm-path", type=str, default="models/lander_vecnorm.pkl", help="Path to VecNormalize stats")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed")
    parser.add_argument("--notes", type=str, default="", help="Custom notes")
    return parser.parse_args()


def main():
    args = parse_args()
    regime = "robust" if args.robust else "clean"

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file {args.model_path} not found.")
        return

    print(f"--- Running {regime.upper()} SB3 Reference Evaluation ---")
    print(f"Model: {args.model_path}")
    print(f"VecNormalize: {args.vecnorm_path}")
    print(f"Episodes: {args.episodes}")

    engine = PolicyEngine(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path if os.path.exists(args.vecnorm_path or "") else None,
        device="cpu",
    )
    evaluator = Evaluator(engine)
    metrics = evaluator.evaluate(regime=regime, episodes=args.episodes, seed=args.seed)

    os.makedirs("eval_results", exist_ok=True)
    results = {
        f"{regime}_mean": metrics.mean_reward,
        f"{regime}_std": metrics.std_reward,
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