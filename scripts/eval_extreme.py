import os
import json
import numpy as np
import gymnasium as gym
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import warnings

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model in extreme conditions")
    parser.add_argument("--model-path", type=str, default="models/best_model.zip", help="Path to the model file")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to VecNormalize stats")
    return parser.parse_args()

# Create extreme environment (max DR, high wind probability)
def extreme_env(env_id, vecnorm_path):
    try:
        # Create base environment with extreme wind parameters
        try:
            # Try with newer parameter names
            env = gym.make(env_id, render_mode=None, 
                          enable_wind=True,
                          wind_power=25.0)  # increased wind power
        except TypeError:
            # Try with older parameter names
            env = gym.make(env_id, render_mode=None, 
                          enable_wind=True,
                          wind_power=25.0)
            warnings.warn("Using older wind parameter format for LunarLander")
    except gym.error.NameNotFound:
        fallback_env_id = "LunarLander-v2"
        print(f"Warning: {env_id} not found, falling back to {fallback_env_id}")
        try:
            env = gym.make(fallback_env_id, render_mode=None,
                          enable_wind=True,
                          wind_power=25.0)
        except TypeError:
            # Try with older parameter names
            env = gym.make(fallback_env_id, render_mode=None, 
                          enable_wind=True,
                          wind_power=25.0)
            warnings.warn("Using older wind parameter format for LunarLander")
    
    # Wrap in DummyVecEnv for VecNormalize compatibility
    env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize stats
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        # Disable training and reward normalization, keep observation normalization
        env.training = False
        env.norm_reward = False
        env.norm_obs = True
    elif vecnorm_path:
        print(f"Warning: VecNormalize stats file {vecnorm_path} not found")
    
    return env

def main():
    args = parse_args()
    
    # Create extreme environment
    env = extreme_env(args.env_id, args.vecnorm_path)
    
    # Load the model
    model = PPO.load(args.model_path)
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.episodes, deterministic=True)
    
    # Print results
    results = {
        "extreme_mean": float(mean_reward),
        "extreme_std": float(std_reward),
        "episodes": args.episodes
    }
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()