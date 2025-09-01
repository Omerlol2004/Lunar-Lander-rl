#!/usr/bin/env python
"""
Playback script for Lunar Lander model demonstration.
This script loads the best model and generates videos for clean and robust environments.
"""

import argparse
import os
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import imageio

def make_env(robust=False, extreme=False):
    """Create a LunarLander environment with optional robust settings."""
    from gym.envs.box2d.lunar_lander import LunarLander
    
    def _init():
        env = gym.make("LunarLander-v3")
        if robust or extreme:
            # Apply domain randomization
            env.unwrapped.gravity = -10.0 * (1.0 + 0.2 * (np.random.rand() - 0.5))
            env.unwrapped.engine_power = 13.0 * (1.0 + 0.2 * (np.random.rand() - 0.5))
            
            # Apply wind disturbance
            wind_power = 0.02 if robust else 0.05  # Higher for extreme
            env = LunarWindDisturbance(env, wind_power=wind_power)
        return env
    
    return _init

class LunarWindDisturbance(gym.Wrapper):
    """Apply random wind forces to the lander."""
    def __init__(self, env, wind_power=0.02):
        super(LunarWindDisturbance, self).__init__(env)
        self.wind_power = wind_power
        
    def step(self, action):
        # Apply random wind force
        self.env.unwrapped.lander.ApplyForceToCenter(
            (self.wind_power * (np.random.rand() * 2 - 1), 0),
            True
        )
        return self.env.step(action)

def generate_video(model_path, vecnorm_path, output_path, episodes=1, robust=False, extreme=False):
    """Generate a video of the model's performance."""
    # Create environment
    env_fn = make_env(robust=robust, extreme=extreme)
    env = DummyVecEnv([env_fn])
    
    # Load VecNormalize statistics
    vec_normalize = VecNormalize.load(vecnorm_path, env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=vec_normalize)
    
    # Setup for video recording
    frames = []
    total_reward = 0
    
    for episode in range(episodes):
        obs = vec_normalize.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = vec_normalize.step(action)
            episode_reward += reward[0]
            
            # Render and capture frame
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        
        total_reward += episode_reward
        print(f"Episode {episode+1}/{episodes} completed")
    
    # Save video
    print(f"Total frames: {len(frames)}")
    print(f"First frame shape: {frames[0].shape}")
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")
    print(f"Average reward: {total_reward / episodes:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Generate videos of Lunar Lander model performance")
    parser.add_argument("--model", type=str, default="models/best_model.zip", help="Path to model file")
    parser.add_argument("--vecnorm", type=str, default="models/lander_vecnorm.pkl", help="Path to VecNormalize file")
    parser.add_argument("--output-dir", type=str, default="videos", help="Output directory for videos")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate videos for clean and robust environments
    print("\nGenerating clean environment video...")
    generate_video(
        args.model, 
        args.vecnorm, 
        os.path.join(args.output_dir, "lander_clean.gif"), 
        episodes=args.episodes
    )
    
    print("\nGenerating robust environment video...")
    generate_video(
        args.model, 
        args.vecnorm, 
        os.path.join(args.output_dir, "lander_robust.gif"), 
        episodes=args.episodes, 
        robust=True
    )
    
    print("\nDone! Videos generated successfully.")

if __name__ == "__main__":
    main()