# LunarLander RL Pipeline Runbook

## Quick Start Guide

### Which model do I use?

For general use, always use `models/best_model.zip` together with `models/lander_vecnorm.pkl`. This hybrid-trained model works well in both clean and robust environments.

### How do I evaluate clean vs robust?

```bash
# Evaluate in clean environment (100 episodes)
python scripts/eval_sb3_ref.py --model-path models/best_model.zip --vecnorm-path models/lander_vecnorm.pkl --episodes 100 --env-id LunarLander-v3

# Evaluate in robust environment (100 episodes)
python scripts/eval_sb3_ref.py --model-path models/best_model.zip --robust --vecnorm-path models/lander_vecnorm.pkl --episodes 100 --env-id LunarLander-v3

# Evaluate in extreme environment (100 episodes)
python scripts/eval_extreme.py --model-path models/best_model.zip --vecnorm-path models/lander_vecnorm.pkl --episodes 100 --env-id LunarLander-v3
```

### Expected Performance Metrics (±10-15% tolerance)

- Clean (100 episodes): ~267 ± 32 (acceptable range: 227-307)
- Robust p=0.02 (100 episodes): ~236 ± 73 (acceptable range: 200-271)
- Extreme p=0.05 (100 episodes): ~172 ± 121 (acceptable range: 146-198)

If any score falls outside these ranges, flag as "performance regression".

This document contains the canonical commands for training, evaluation, and reflection in our standardized RL pipeline.

## Prerequisites

- Python 3.8+
- Conda environment with required packages

```bash
# Activate the conda environment
conda activate rl-lander
```

## Training Commands

### Stage A: Clean Baseline

```bash
# Train clean baseline with seed 0
conda run -n rl-lander python scripts/train.py --seed 0 --config configs/lunarlander_ppo_clean.yaml --out-dir runs/clean_seed_0

# Train clean baseline with seed 1
conda run -n rl-lander python scripts/train.py --seed 1 --config configs/lunarlander_ppo_clean.yaml --out-dir runs/clean_seed_1

# Train clean baseline with seed 2
conda run -n rl-lander python scripts/train.py --seed 2 --config configs/lunarlander_ppo_clean.yaml --out-dir runs/clean_seed_2
```

### Reflection and Retry

After training, evaluate each seed to determine the best performer:

```bash
# Evaluate seed 0
conda run -n rl-lander python scripts/eval_sb3_ref.py --model-path runs/clean_seed_0/final.zip --episodes 100

# Evaluate seed 1
conda run -n rl-lander python scripts/eval_sb3_ref.py --model-path runs/clean_seed_1/final.zip --episodes 100

# Evaluate seed 2
conda run -n rl-lander python scripts/eval_sb3_ref.py --model-path runs/clean_seed_2/final.zip --episodes 100
```

Select the best seed and copy it to the canonical location:

```bash
# Copy the best model to canonical location
copy runs/clean_seed_X/final.zip models/best_model_clean.zip
```

### Stage B: Robust Model

Only proceed if Stage A passed the gate (mean reward ≥ 250):

```bash
# Fine-tune from clean model with domain randomization and wind
python scripts/train.py --seed 0 --config configs/lunarlander_ppo_robust.yaml --out-dir runs/robust_training_seed_0 --model-path models/best_model_clean.zip
```

After training, copy the model and VecNormalize stats to canonical locations:

```bash
# Copy the robust model to canonical location
copy runs/robust_training_seed_0/final.zip models/best_model.zip
copy runs/robust_training_seed_0/vecnormalize.pkl models/lander_vecnorm.pkl
```

## Evaluation Commands

### Clean Environment

```bash
# Evaluate clean model on clean environment
conda run -n rl-lander python scripts/eval_sb3_ref.py --model-path models/best_model_clean.zip --episodes 100 --env-id LunarLander-v3

# Evaluate robust model on clean environment
python scripts/eval_sb3_ref.py --model-path models/best_model.zip --vecnorm-path models/lander_vecnorm.pkl --episodes 100 --env-id LunarLander-v3

# Generate video for clean model
conda run -n rl-lander python scripts/generate_video.py --model-path models/best_model_clean.zip --output videos/lander_clean.gif --episodes 3 --env-id LunarLander-v3
```

### Robust Environment

```bash
# Evaluate robust model on environment with DR and wind
python scripts/eval_sb3_ref.py --model-path models/best_model.zip --robust --vecnorm-path models/lander_vecnorm.pkl --episodes 100 --env-id LunarLander-v3

# Generate video for robust model
python scripts/generate_video.py --model-path models/best_model.zip --vecnorm-path models/lander_vecnorm.pkl --robust --output videos/lander_robust.gif --episodes 3 --env-id LunarLander-v3
```

### Extreme Environment

```bash
# Evaluate robust model on extreme environment with higher wind probability
python scripts/eval_extreme.py --model-path models/best_model.zip --vecnorm-path models/lander_vecnorm.pkl --episodes 100 --env-id LunarLander-v3

# Generate video for extreme conditions
python scripts/generate_extreme_video.py --model-path models/best_model.zip --vecnorm-path models/lander_vecnorm.pkl --output videos/lander_extreme.gif --episodes 3 --env-id LunarLander-v3
```

## Canonical File Structure

```
├── configs/
│   ├── lunarlander_ppo_clean.yaml   # Stage A config
│   └── lunarlander_ppo_robust.yaml  # Stage B config
├── scripts/
│   ├── train.py                     # Training entrypoint
│   ├── eval_sb3_ref.py              # Reference evaluator
│   ├── eval_extreme.py              # Extreme environment evaluator
│   ├── generate_video.py            # Video generation
│   └── generate_extreme_video.py    # Extreme video generation
├── models/
│   ├── best_model_clean.zip         # Best Stage A model
│   ├── best_model.zip               # Best Stage B model (robust)
│   └── lander_vecnorm.pkl           # VecNormalize stats for robust model
├── videos/
│   ├── lander_clean.gif             # Clean environment video
│   ├── lander_robust.gif            # Robust environment video
│   └── lander_extreme.gif           # Extreme environment video
├── eval_results/
│   └── receipt.json                 # Evaluation results
├── runs/                            # Training run directories
├── logs/                            # Training metrics
├── eval_results/                    # Evaluation results
│   └── receipt.json                 # Final receipt
├── videos/                          # Generated videos
│   ├── lander_clean.gif             # Clean model video
│   └── lander_robust.gif            # Robust model video (if applicable)
└── RUNBOOK.md                       # This file
```

## Maintenance

### Repository Verification

To verify the repository is in a consistent state, run:

```bash
# Using make
make verify

# Or using the verification script directly
bash scripts/verify.sh
```

This will check that:
- All canonical files are present (best_model.zip, lander_vecnorm.pkl, best_model_clean.zip, receipt.json)
- Only allowed files exist in videos/ directory
- No extra JSON files exist in eval_results/ directory

### Pre-commit Checks

A pre-commit hook is installed that will prevent commits if:
- Multiple JSON files exist in eval_results/ (only receipt.json should be present)
- Required model files (best_model.zip, lander_vecnorm.pkl) are missing
- Empty directories are present in the repository

### Continuous Integration

A GitHub Actions workflow automatically verifies the repository and runs performance tests:
- Triggers: On pull requests, pushes to main, and weekly (Monday 6:00 AM)
- Checks: Repository structure verification and performance evaluation
- Performance ranges: Enforces the tolerances specified in "Expected Performance Metrics" section
- Artifacts: Evaluation videos are uploaded as workflow artifacts

To view CI results and artifacts:
1. Check the status badge in the README
2. Navigate to the Actions tab in the GitHub repository
3. Select the most recent workflow run
4. Download artifacts from the "Artifacts" section

**Important:** For proper repository protection, enable branch protection rules that require status checks to pass before merging. See `BRANCH_PROTECTION.md` for detailed configuration instructions.

### Archived Files

Non-essential files are archived in the `archive/` directory, organized by type:
- `archive/models/` - Additional model files not needed for standard operation
- `archive/videos/` - Additional video files not needed for standard operation
- `archive/eval_results/` - Historical evaluation results
- `archive/runs/` - Historical training runs (only two most recent runs are kept in the main directory)
- `archive/scripts/` - Additional scripts not needed for standard operation