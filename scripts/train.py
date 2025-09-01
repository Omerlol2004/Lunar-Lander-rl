"""
Train LunarLander with SB3-style PPO.
This script implements the exact hyperparameters specified in the requirements.
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import torch
import yaml
from pathlib import Path
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# Define a wrapper for domain randomization
class LunarLanderDRWrapper(gym.Wrapper):
    def __init__(self, env, dr_config=None):
        super().__init__(env)
        self.dr_config = dr_config if dr_config else {}

    def reset(self, **kwargs):
        if self.dr_config:
            # Apply domain randomization to the environment
            self.env.unwrapped.randomize(self.dr_config)
        return self.env.reset(**kwargs)

# Function to create environment
def make_env(env_id, rank, seed=0, use_wind=False, wind_prob=0.0, use_dr=False, dr_config=None):
    def _init():
        env_kwargs = {}
        if use_wind:
            env_kwargs['enable_wind'] = True
            env_kwargs['wind_power'] = 15.0
            env_kwargs['turbulence_power'] = 1.5
            
        try:
            # Try to create the specified environment
            env = gym.make(env_id, **env_kwargs)
        except (gym.error.NameNotFound, gym.error.VersionNotFound):
            print(f"Warning: {env_id} not found, falling back to LunarLander-v2")
            env = gym.make("LunarLander-v2", **env_kwargs)
        
        if use_dr:
            env = LunarLanderDRWrapper(env, dr_config)

        # Set the seed
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

# Function to create vectorized environment
def create_vec_env(env_id, n_envs, seed, use_vecnorm=False, use_wind=False, wind_prob=0.0, use_dr=False, dr_config=None):
    # Use DummyVecEnv for better Windows compatibility
    env = DummyVecEnv([make_env(env_id, i, seed, use_wind, wind_prob, use_dr, dr_config) for i in range(n_envs)])
    
    # Apply VecNormalize if specified
    if use_vecnorm:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    return env

# Linear learning rate schedule
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_config, total_timesteps, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum_config = curriculum_config
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        for param in self.curriculum_config:
            name = param['name']
            start = param['start']
            end = param['end']
            schedule = param.get('schedule', 'linear')

            if schedule == 'linear':
                value = start + (end - start) * progress
            else:
                value = start # Default to start value if schedule is not linear

            # Apply the curriculum to the environment
            for env in self.training_env.envs:
                if hasattr(env.unwrapped, name):
                    setattr(env.unwrapped, name, value)
        return True

# Function to train a model with a specific seed
def train(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up environment ID and check if it exists
    env_id = config.get('env_id', 'LunarLander-v3')
    try:
        test_env = gym.make(env_id)
        # Verify action space is discrete with 4 actions
        assert hasattr(test_env.action_space, "n"), f"Expected discrete action space, got {test_env.action_space}"
        assert test_env.action_space.n == 4, f"Expected 4 discrete actions, got {test_env.action_space.n}"
        print(f"Environment: {env_id}")
        print(f"Action space: {test_env.action_space}")
        print(f"Observation space: {test_env.observation_space}")
        test_env.close()
    except (gym.error.NameNotFound, gym.error.VersionNotFound):
        print(f"Warning: {env_id} not found, falling back to LunarLander-v2")
        env_id = "LunarLander-v2"
        test_env = gym.make(env_id)
        assert hasattr(test_env.action_space, "n"), f"Expected discrete action space, got {test_env.action_space}"
        assert test_env.action_space.n == 4, f"Expected 4 discrete actions, got {test_env.action_space.n}"
        print(f"Environment: {env_id}")
        print(f"Action space: {test_env.action_space}")
        print(f"Observation space: {test_env.observation_space}")
        test_env.close()
    
    # Create output directory
    if args.out_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", f"{config.get('algo', 'ppo')}_{timestamp}_seed_{args.seed}")
    else:
        out_dir = args.out_dir
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    print(f"Created output directory: {out_dir}")
    
    # Set up training parameters from config
    n_envs = config.get('n_envs', 16)
    n_steps = config.get('n_steps', 2048)
    batch_size = config.get('batch_size', 64)
    n_epochs = config.get('n_epochs', 10)
    gamma = config.get('gamma', 0.99)
    gae_lambda = config.get('gae_lambda', 0.95)
    clip_range = config.get('clip_range', 0.2)
    ent_coef = config.get('ent_coef', 0.01)
    vf_coef = config.get('vf_coef', 0.5)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    learning_rate = float(config.get('learning_rate', 3e-4))
    total_timesteps = config.get('timesteps', 2000000)
    use_vecnormalize = config.get('vecnormalize', False)
    
    # Domain randomization and wind settings
    use_curriculum = config.get('use_curriculum', False)
    use_randomization = config.get('use_randomization', False)
    wind_config = config.get('wind', {})
    wind_prob = wind_config.get('p', 0.0)
    curriculum_config = config.get('curriculum', None)
    
    # Create environment
    env = create_vec_env(env_id, n_envs, args.seed, use_vecnormalize, 
                        use_wind=wind_prob > 0 or use_curriculum, wind_prob=wind_prob,
                        use_dr=use_randomization, dr_config=config.get('dr_config', None))
    
    # Set up policy network architecture
    policy_kwargs_config = config.get('policy_kwargs', {})
    net_arch = policy_kwargs_config.get('net_arch', [64, 64])
    activation_fn_name = policy_kwargs_config.get('activation_fn', 'tanh')
    
    # Convert activation function name to actual function
    if activation_fn_name.lower() == 'tanh':
        activation_fn = torch.nn.Tanh
    elif activation_fn_name.lower() == 'relu':
        activation_fn = torch.nn.ReLU
    else:
        activation_fn = torch.nn.Tanh
    
    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation_fn
    }
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(out_dir, "checkpoints"),
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=use_vecnormalize,
    )
    callbacks = [checkpoint_callback]
    if use_curriculum and curriculum_config:
        callbacks.append(CurriculumCallback(curriculum_config, total_timesteps))

    # Learning rate schedule
    lr_schedule_type = config.get('lr_schedule', 'constant')
    if lr_schedule_type == 'linear':
        lr = linear_schedule(learning_rate)
    else:
        lr = learning_rate
    
    # Create or load model
    if args.load_from and os.path.exists(args.load_from):
        print(f"Loading model from {args.load_from}")
        model = PPO.load(
            args.load_from,
            env=env,
            device=config.get('device', 'auto'),
            learning_rate=lr,
        )
        print(f"Model loaded. Current timesteps: {model.num_timesteps}")
    else:
        # Create model
        model = PPO(
            config.get('policy', 'MlpPolicy'),
            env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=config.get('device', 'auto')
        )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=False
    )
    
    # Save the final model
    final_model_path = os.path.join(out_dir, "final.zip")
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save VecNormalize stats if used
    if use_vecnormalize:
        vecnorm_path = os.path.join(out_dir, "vecnormalize.pkl")
        env.save(vecnorm_path)
        print(f"Saved VecNormalize stats to {vecnorm_path}")
    
    # Close environment
    env.close()
    
    return final_model_path, out_dir

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train LunarLander with SB3-style PPO")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--load-from", type=str, default=None, help="Path to a model to continue training from")
    
    args = parser.parse_args()
    
    # Train the model
    print(f"=== Training with seed {args.seed} ===")
    model_path, run_dir = train(args)
    print(f"=== Training completed successfully ===")
    print(f"Model saved to: {model_path}")
    print(f"Run directory: {run_dir}")

if __name__ == "__main__":
    main()