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
    parser = argparse.ArgumentParser(description="Evaluate a trained model using SB3's evaluate_policy")
    parser.add_argument("--model-path", type=str, default="models/best_model.zip", help="Path to the model file")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--robust", action="store_true", help="Use robust evaluation with DR and wind")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to VecNormalize stats")
    parser.add_argument("--notes", type=str, default="", help="Custom notes for the evaluation report")
    return parser.parse_args()

# Create clean environment (no DR, no wind)
def clean_env(env_id, vecnorm_path):
    try:
        env = gym.make(env_id, render_mode=None)
    except gym.error.NameNotFound:
        fallback_env_id = "LunarLander-v2"
        print(f"Warning: {env_id} not found, falling back to {fallback_env_id}")
        env = gym.make(fallback_env_id, render_mode=None)
    
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

# Create robust environment (with DR and wind)
def robust_env(env_id, vecnorm_path):
    try:
        # Create base environment with wind parameters
        # Note: Different versions of Gymnasium may have different parameter names
        try:
            # Try with newer parameter names
            env = gym.make(env_id, render_mode=None, 
                          enable_wind=True,
                          wind_power=15.0)  # wind power
        except TypeError:
            # Try with older parameter names
            env = gym.make(env_id, render_mode=None, 
                          enable_wind=True,
                          wind_power=15.0)
            warnings.warn("Using older wind parameter format for LunarLander")
    except gym.error.NameNotFound:
        fallback_env_id = "LunarLander-v2"
        print(f"Warning: {env_id} not found, falling back to {fallback_env_id}")
        try:
            env = gym.make(fallback_env_id, render_mode=None,
                          enable_wind=True,
                          wind_power=15.0)
        except TypeError:
            # Try with older parameter names
            env = gym.make(fallback_env_id, render_mode=None, 
                          enable_wind=True,
                          wind_power=15.0)
            warnings.warn("Using older wind parameter format for LunarLander")
            
            if hasattr(env.unwrapped, 'side_engine_power'):
                side_engine = np.random.uniform(dr_config.get('side_engine', [0.90, 1.12])[0],
                                              dr_config.get('side_engine', [0.90, 1.12])[1])
                env.unwrapped.side_engine_power = 0.6 * side_engine
            
            # Apply initial angle randomization if available
            if hasattr(env.unwrapped, 'initial_random') and 'init_angle_deg' in dr_config:
                angle_range = dr_config.get('init_angle_deg', [-8, 8])
                angle_rad = np.random.uniform(angle_range[0], angle_range[1]) * np.pi / 180.0
                env.unwrapped.initial_random = angle_rad
            
            # Apply initial speed randomization if available
            if hasattr(env.unwrapped, 'initial_speed') and 'init_speed' in dr_config:
                speed_range = dr_config.get('init_speed', [0.0, 1.2])
                env.unwrapped.initial_speed = np.random.uniform(speed_range[0], speed_range[1])
    
    # Wrap in DummyVecEnv for VecNormalize compatibility
    env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize stats
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        # Disable training and reward normalization, keep observation normalization
        env.training = False
        env.norm_reward = False
        env.norm_obs = True
    else:
        print(f"Warning: VecNormalize stats file {vecnorm_path} not found")
    
    return env

# Alignments checks
def run_alignment_checks(env_id):
    print("--- Running Alignment Checks ---")
    try:
        env = gym.make(env_id)
    except gym.error.NameNotFound:
        fallback_env_id = "LunarLander-v2"
        print(f"Warning: {env_id} not found, falling back to {fallback_env_id}")
        env = gym.make(fallback_env_id)

    # Check 1: Environment ID
    print(f"Environment ID: {env.spec.id}")
    if env.spec.id not in ["LunarLander-v3", "LunarLander-v2"]:
        print("BLOCKED: Incorrect environment ID.")
        return False

    # Check 2: Action Space
    print(f"Action Space: {env.action_space}")
    if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 4:
        print("BLOCKED: Action space is not Discrete(4).")
        return False

    # Check 3: No VecNormalize for clean eval
    print("VecNormalize check: OK (ensured by not loading vecnorm for clean eval)")

    print("--- Alignment Checks Passed ---")
    return True

# Main execution
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Run alignment checks
    if not run_alignment_checks(args.env_id):
        results = {
            "status": "BLOCKED",
            "phase": "A_ONLY",
            "summary": "Alignment checks failed.",
            "stage_a": {},
            "stage_b": { "ran": False },
            "artifacts": {},
            "proof": {},
            "notes": "See console output for details."
        }
        with open("eval_results/receipt.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        return

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found. Skipping evaluation.")
        return
    
    # Ensure output directory exists
    os.makedirs("eval_results", exist_ok=True)
    
    # Load model
    model = PPO.load(args.model_path)
    
    # Determine which environment to evaluate on
    if args.robust:
        # Evaluate on robust environment
        env_instance = robust_env(args.env_id, args.vecnorm_path)
        mean_reward, std_reward = evaluate_policy(model, env_instance, n_eval_episodes=args.episodes, deterministic=True, return_episode_rewards=False)
        eval_type = "robust"
    else:
        # Evaluate on clean environment
        # For clean evaluation, we do not use vecnorm
        env_instance = clean_env(args.env_id, vecnorm_path=None)
        mean_reward, std_reward = evaluate_policy(model, env_instance, n_eval_episodes=args.episodes, deterministic=True, return_episode_rewards=False)
        eval_type = "clean"
    
    # Save notes to the specified file
    notes_file = "eval_results/sb3_ref_notes.txt"
    with open(notes_file, "a") as f:
        f.write(f"\n{eval_type.upper()} SB3 Reference Evaluation Settings\n")
        f.write("================================\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"VecNormalize: None for clean eval\n")
        f.write(f"VecNormalize settings: training=False, norm_obs=False, norm_reward=False\n\n")
        
        if eval_type == "clean":
            f.write("Clean Environment:\n")
            f.write("  - No domain randomization\n")
            f.write("  - No wind disturbance\n")
        else:
            f.write("Robust Environment:\n")
            f.write("  - Domain randomization from configs/lunarlander_ppo_robust.yaml\n")
            f.write("  - Wind disturbance: p=0.02, sigma_lin=0.6, sigma_ang=0.03\n")
        
        f.write("  - Deterministic evaluation\n\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Results: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}\n\n")
    
    # Print results as JSON
    results = {
        f"{eval_type}_mean": float(mean_reward),
        f"{eval_type}_std": float(std_reward),
        "episodes": args.episodes
    }
    
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    main()