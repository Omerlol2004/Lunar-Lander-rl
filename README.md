# Lunar Lander Reinforcement Learning

This project trains and packages a PPO reinforcement-learning agent for `LunarLander-v3`. The final release includes trained Stable-Baselines3 models, VecNormalize statistics, evaluation receipts, demo GIFs, and operational documentation for reproducing clean, robust, and extreme-condition evaluations.

The main objective is not only to land the spacecraft in the standard environment, but also to preserve useful performance under wind and domain-randomized conditions.

## Table of Contents

- [Project Objective](#project-objective)
- [Demo](#demo)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Model Artifacts](#model-artifacts)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [Training Workflow](#training-workflow)
- [Repository Structure](#repository-structure)
- [Current Status and Limitations](#current-status-and-limitations)
- [Documentation](#documentation)
- [License](#license)

## Project Objective

The goal is to build a Lunar Lander agent that can:

1. Learn a high-scoring landing policy in the standard `LunarLander-v3` environment.
2. Fine-tune that policy for robustness against wind and environment variation.
3. Preserve clean-environment performance after robust fine-tuning.
4. Package the final model with the normalization statistics required for inference.
5. Provide reproducible evaluation commands and visual demonstrations.

## Demo

| Clean Environment | Robust Environment | Extreme Environment |
| --- | --- | --- |
| ![Clean lunar lander demo](videos/lander_clean.gif) | ![Robust lunar lander demo](videos/lander_robust.gif) | ![Extreme lunar lander demo](videos/lander_extreme.gif) |

The three demos show the same trained agent family under progressively harder conditions: standard dynamics, wind/domain-randomized dynamics, and stronger disturbance settings.

## Key Results

Final reported evaluation metrics are based on 100 deterministic evaluation episodes.

| Evaluation Condition | Mean Reward | Std. Dev. | Notes |
| --- | ---: | ---: | --- |
| Clean | 267.15 | 31.77 | Standard LunarLander-v3 environment |
| Robust | 235.58 | 73.29 | Wind/domain-randomized evaluation |
| Extreme | 171.79 | 121.13 | Stronger disturbance setting |

Training receipt summary:

| Stage | Result |
| --- | --- |
| Stage A clean baseline | 278.41 +/- 18.65 |
| Best clean seed | 0 |
| Stage A training steps | 5,029,888 |
| Stage B hybrid fine-tuned model | 25% clean / 75% robust |
| Total training steps | 16,397,056 |
| Final status | DONE |

The canonical evaluation record is stored in [`eval_results/receipt.json`](eval_results/receipt.json).

## Methodology

The project uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3 with a small MLP policy.

| Area | Configuration |
| --- | --- |
| Environment | `LunarLander-v3` |
| Algorithm | PPO |
| Policy | `MlpPolicy` |
| Network | `[64, 64]` MLP |
| Activation | `tanh` |
| Discount factor | `gamma = 0.99` |
| GAE lambda | `0.95` |
| PPO clip range | `0.2` |
| Entropy coefficient | `0.01` |
| Vectorized envs | 16 |
| Clean config | [`configs/lunarlander_ppo_clean.yaml`](configs/lunarlander_ppo_clean.yaml) |
| Robust config | [`configs/lunarlander_ppo_robust.yaml`](configs/lunarlander_ppo_robust.yaml) |

The training process is split into two stages:

1. **Stage A: Clean baseline**
   - Train PPO in the standard environment.
   - Evaluate multiple seeds.
   - Keep the best clean checkpoint as a baseline.

2. **Stage B: Robust fine-tuning**
   - Continue from the clean model.
   - Enable wind, curriculum settings, and domain randomization.
   - Save the final robust model together with VecNormalize statistics.

## Model Artifacts

| File | Purpose |
| --- | --- |
| `models/best_model.zip` | Final hybrid model recommended for general use |
| `models/lander_vecnorm.pkl` | Required VecNormalize statistics for the final model |
| `models/best_model_clean.zip` | Clean-environment baseline retained for comparison |
| `models/checkpoints/` | Recent training checkpoints |
| `videos/lander_clean.gif` | Clean-environment demo |
| `videos/lander_robust.gif` | Robust-environment demo |
| `videos/lander_extreme.gif` | Extreme-environment demo |

Important: for general use, load `models/best_model.zip` together with `models/lander_vecnorm.pkl`. The normalization statistics are part of the trained policy pipeline and should not be skipped.

## Installation

Clone the repository:

```bash
git clone https://github.com/Omerlol2004/Lunar-Lander-rl.git
cd Lunar-Lander-rl
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate lunar_lander_rl
```

Or install from `requirements.txt` in an existing Python environment:

```bash
pip install -r requirements.txt
```

Verify the environment:

```bash
python check_env.py
```

## Quick Start

Run a short playback demo:

```bash
python scripts/playback.py --episodes 1
```

Load the final model programmatically:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: gym.make("LunarLander-v3")])

vec_env = VecNormalize.load("models/lander_vecnorm.pkl", env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("models/best_model.zip", env=vec_env)

obs = vec_env.reset()
done = [False]
episode_reward = 0.0

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    episode_reward += float(reward[0])

print(f"Episode reward: {episode_reward:.2f}")
```

## Evaluation

Evaluate the final model in the clean environment:

```bash
python scripts/eval_sb3_ref.py \
  --model-path models/best_model.zip \
  --vecnorm-path models/lander_vecnorm.pkl \
  --episodes 100 \
  --env-id LunarLander-v3
```

Evaluate under robust wind/domain-randomized conditions:

```bash
python scripts/eval_sb3_ref.py \
  --model-path models/best_model.zip \
  --robust \
  --vecnorm-path models/lander_vecnorm.pkl \
  --episodes 100 \
  --env-id LunarLander-v3
```

Evaluate the extreme disturbance setting:

```bash
python scripts/eval_extreme.py \
  --model-path models/best_model.zip \
  --vecnorm-path models/lander_vecnorm.pkl \
  --episodes 100 \
  --env-id LunarLander-v3
```

Expected score ranges are documented in [`RUNBOOK.md`](RUNBOOK.md). Results that fall well outside those ranges should be treated as potential regressions.

## Training Workflow

Train a clean baseline:

```bash
python scripts/train.py \
  --seed 0 \
  --config configs/lunarlander_ppo_clean.yaml \
  --out-dir runs/clean_seed_0
```

Fine-tune for robust conditions:

```bash
python scripts/train.py \
  --seed 0 \
  --config configs/lunarlander_ppo_robust.yaml \
  --out-dir runs/robust_training_seed_0 \
  --load-from models/best_model_clean.zip
```

After training, copy the selected artifacts into the canonical paths:

```text
models/best_model.zip
models/best_model_clean.zip
models/lander_vecnorm.pkl
```

## Repository Structure

```text
.
|-- README.md
|-- MODEL_CARD.md
|-- RUNBOOK.md
|-- CHANGELOG.md
|-- environment.yml
|-- requirements.txt
|-- configs/
|   |-- lunarlander_ppo_clean.yaml
|   `-- lunarlander_ppo_robust.yaml
|-- scripts/
|   |-- train.py
|   |-- eval_sb3_ref.py
|   |-- eval_extreme.py
|   |-- playback.py
|   |-- generate_video.py
|   `-- generate_extreme_video.py
|-- models/
|   |-- best_model.zip
|   |-- best_model_clean.zip
|   |-- lander_vecnorm.pkl
|   `-- checkpoints/
|-- videos/
|   |-- lander_clean.gif
|   |-- lander_robust.gif
|   `-- lander_extreme.gif
`-- eval_results/
    `-- receipt.json
```

## Current Status and Limitations

Current strengths:

- Final PPO model and normalization statistics are included in the repository.
- Clean, robust, and extreme-condition demos are available as GIFs.
- Evaluation receipt records the final reported metrics.
- The model card and runbook document the intended usage and performance gates.
- The repository includes scripts for training, evaluation, playback, and video generation.

Known limitations:

- Robust performance has higher variance than clean performance, especially under extreme disturbance.
- The final model depends on `lander_vecnorm.pkl`; using the model without the saved normalization state can reduce performance.
- Disturbance handling is tuned to the wind/domain-randomization patterns used here and may not generalize to substantially different physics changes.
- Some historical documentation references archived or CI-related workflows that may need repository-specific setup before use.
- Full training is compute-intensive because the final recipe uses more than 16M total training steps.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) - model details, performance, usage notes, and limitations.
- [`RUNBOOK.md`](RUNBOOK.md) - canonical training, evaluation, verification, and maintenance commands.
- [`CHANGELOG.md`](CHANGELOG.md) - release history.
- [`BRANCH_PROTECTION.md`](BRANCH_PROTECTION.md) - suggested repository protection settings.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

Third-party libraries used by this project include:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [Box2D / pybox2d](https://github.com/pybox2d/pybox2d)
