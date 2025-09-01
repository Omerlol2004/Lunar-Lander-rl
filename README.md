# Lunar Lander Reinforcement Learning Project

[![Repository Verification and Performance Testing](https://github.com/username/Reinforcement-learning/actions/workflows/verify.yml/badge.svg)](https://github.com/username/Reinforcement-learning/actions/workflows/verify.yml)

## 🚀 LunarLander RL v1.1 Release

- **Clean**: 267.15 ± 31.77 (100 eps)
- **Robust (p=0.02)**: 235.58 ± 73.29 (100 eps)
- **Extreme (p=0.05)**: 171.79 ± 121.13 (100 eps)

This bundle includes models, VecNormalize stats, videos, a single receipt, and a one-page RUNBOOK.

## Quick Start

### How to Use

```python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create environment
def make_env():
    return lambda: gym.make("LunarLander-v3")

env = DummyVecEnv([make_env()])

# Load VecNormalize statistics (IMPORTANT)
vec_normalize = VecNormalize.load("models/lander_vecnorm.pkl", env)
vec_normalize.training = False
vec_normalize.norm_reward = False

# Load model
model = PPO.load("models/best_model.zip", env=vec_normalize)

# Run inference
obs = vec_normalize.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_normalize.step(action)
    total_reward += reward[0]
    
    # Optional: render
    vec_normalize.render()

print(f"Episode reward: {total_reward}")
```

### Model Usage Guidelines

- **General use**: Always use `models/best_model.zip` with `models/lander_vecnorm.pkl`
- **Clean model**: `models/best_model_clean.zip` is only for calm/no-disturbance demos (historical baseline)

### Quick Demo

```bash
# Run a quick demo in clean environment
python scripts/playback.py --env clean --episodes 1

# Run a quick demo in robust environment
python scripts/playback.py --env robust --episodes 1
```

## How to Cite

If you use this model in your research, please cite it as:

```
Author. (2025). Lunar Lander Reinforcement Learning Project: A robust PPO implementation for the LunarLander-v3 environment. GitHub. https://github.com/username/Reinforcement-learning
```

## Documentation

- **RUNBOOK.md**: Quick start guide and operational commands
- **MODEL_CARD.md**: Detailed model information and performance metrics
- **CHANGELOG.md**: Version history and release notes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Third-party Licenses

This project uses the following open-source libraries:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - MIT License
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - MIT License
- [Box2D](https://github.com/pybox2d/pybox2d) - zlib License