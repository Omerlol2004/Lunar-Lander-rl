# Lunar Lander Model Card

## Overview

This model is trained to control a lunar lander in the OpenAI Gym LunarLander-v3 environment. It can successfully land the spacecraft in both clean (standard) environments and robust environments with disturbances.

## Model Details

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Architecture**: MLP Policy (ActorCriticPolicy)
- **Training Recipe**:
  - Initial training on clean environments
  - Hybrid fine-tuning with 25% clean and 75% robust environments
  - Seeds: 0 (primary)
  - Total training steps: ~16.4M (including fine-tuning)
  - Early stopping when performance gates were met

## Performance

- **Clean Environment**: 267.15 average reward
- **Robust Environment**: 235.58 average reward
- **Extreme Environment**: 171.79 average reward

## Evaluation Protocol

The model was evaluated across three environment types:
1. **Clean**: Standard LunarLander-v3 environment
2. **Robust**: Environment with domain randomization (gravity, engine power) and wind disturbances
3. **Extreme**: Environment with more severe disturbances

Evaluation metrics are based on average rewards across 100 episodes for each environment type.

## Known Limitations

- Performance may degrade in environments with disturbance patterns significantly different from those seen during training
- The model requires VecNormalize statistics for proper inference, especially in robust environments
- While the hybrid model performs well in both clean and robust environments, specialized models might achieve marginally better performance in specific conditions

## Usage Instructions

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gym

# Create environment
env = gym.make("LunarLander-v3")
env = DummyVecEnv([lambda: env])

# Load VecNormalize statistics (REQUIRED for robust/hybrid model)
vec_normalize = VecNormalize.load("models/lander_vecnorm.pkl", env)
vec_normalize.training = False  # Don't update normalization statistics during inference
vec_normalize.norm_reward = False  # Don't normalize rewards during inference

# Load model
model = PPO.load("models/best_model.zip", env=vec_normalize)

# Run inference
obs = vec_normalize.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_normalize.step(action)
```

## Important Note

Always use `models/best_model.zip` together with `models/lander_vecnorm.pkl` for general use. This hybrid-trained model works well in both clean and robust environments. The `best_model_clean.zip` is kept only as a historical baseline and is not recommended for general use.

Load robust model with VecNormalize (obs-only); eval with training=false, norm_obs=true, norm_reward=false.