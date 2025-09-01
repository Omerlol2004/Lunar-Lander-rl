#!/usr/bin/env python3
"""
Generate videos of trained models in the LunarLander environment.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import imageio

def make_env(env_id, robust=False, wind_prob=0.0):
    """Create environment with optional robust settings."""
    def _init():
        try:
            if robust:
                # Try different parameter names based on Gymnasium version
                try:
                    # Newer versions use turbulence_power
                    env = gym.make(env_id, render_mode="rgb_array", enable_wind=True, wind_power=15.0, turbulence_power=wind_prob)
                except TypeError:
                    try:
                        # Older versions might use wind_probability
                        env = gym.make(env_id, render_mode="rgb_array", enable_wind=True, wind_power=15.0)
                        print("Note: Using enable_wind without probability parameter")
                    except TypeError:
                        # Fallback to basic parameters
                        env = gym.make(env_id, render_mode="rgb_array")
                        print("Warning: Could not enable wind, using standard environment")
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            # Verify render mode is properly set
            print(f"Environment created with render_mode: {env.render_mode}")
        except gym.error.NameNotFound:
            fallback_env_id = "LunarLander-v2"
            print(f"Warning: {env_id} not found, falling back to {fallback_env_id}")
            if robust:
                try:
                    env = gym.make(fallback_env_id, render_mode="rgb_array", enable_wind=True, wind_power=15.0)
                except TypeError:
                    env = gym.make(fallback_env_id, render_mode="rgb_array")
                    print("Warning: Could not enable wind in fallback environment")
            else:
                env = gym.make(fallback_env_id, render_mode="rgb_array")
            # Verify render mode is properly set
            print(f"Environment created with render_mode: {env.render_mode}")
        return env
    return _init

def generate_video(model_path, output_path, env_id="LunarLander-v3", episodes=3, robust=False, vecnorm_path=None):
    """Generate a video of the model in action."""
    # Create environment
    env_fn = make_env(env_id, robust=robust, wind_prob=0.02 if robust else 0.0)
    env = env_fn()
    
    # Create a separate environment for VecNormalize if needed
    vec_env = None
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        # Create a separate environment for normalization
        vec_env = DummyVecEnv([make_env(env_id, robust=robust, wind_prob=0.02 if robust else 0.0)])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.norm_obs = True
        print(f"Loaded VecNormalize stats from {vecnorm_path}")
    elif vecnorm_path is not None:
        print(f"Warning: VecNormalize stats file {vecnorm_path} not found")
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Generate video frames
    frames = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_frames = []
        
        while not done:
            # Get action from model
            if vec_env is not None:
                # For VecNormalize
                try:
                    vec_obs = vec_env.normalize_obs(np.array([obs]))
                    action, _ = model.predict(vec_obs[0], deterministic=True)
                except Exception as e:
                    print(f"Error with VecNormalize: {e}")
                    # Fallback to direct prediction
                    action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render and add to frames
            frame = env.render()
            if frame is None or frame.size == 0:
                print("Warning: Empty frame detected!")
                continue
            print(f"Frame shape: {frame.shape}")
            episode_frames.append(frame)
        
        print(f"Episode {episode+1}/{episodes} completed")
        frames.extend(episode_frames)
    
    # Save as GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if we have valid frames
    if not frames:
        print("Error: No frames were captured!")
        return
    
    print(f"Total frames: {len(frames)}")
    print(f"First frame shape: {frames[0].shape}")
    
    try:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        # Try with lower fps as fallback
        try:
            print("Trying with lower fps...")
            imageio.mimsave(output_path, frames, fps=15)
            print(f"Video saved to {output_path} with lower fps")
        except Exception as e2:
            print(f"Failed to save video: {e2}")
    
    # Close environment
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Generate videos of trained models")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--env-id", type=str, default="LunarLander-v3", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--robust", action="store_true", help="Use robust evaluation with wind")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to VecNormalize stats")
    
    args = parser.parse_args()
    generate_video(
        args.model_path, 
        args.output, 
        args.env_id, 
        args.episodes, 
        args.robust, 
        args.vecnorm_path
    )

if __name__ == "__main__":
    main()