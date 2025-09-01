#!/bin/bash
# Comprehensive script to run the entire pipeline: fine-tune, evaluate, generate videos and artifacts

set -e  # Exit on error

echo "=== Lunar Lander RL Pipeline ==="
echo ""

# Step 1: Fine-tune the model with hybrid approach
echo "Step 1: Fine-tuning model with hybrid approach (25% clean, 75% robust)..."
python scripts/hybrid_finetune.py \
  --model models/best_model.zip \
  --vecnorm models/lander_vecnorm.pkl \
  --output-dir models \
  --seed 0

echo "Fine-tuning complete!"
echo ""

# Step 2: Generate videos for all environments
echo "Step 2: Generating demonstration videos..."
python scripts/generate_video.py \
  --model models/best_model.zip \
  --vecnorm models/lander_vecnorm.pkl \
  --env-id LunarLander-v3 \
  --output videos/lander_clean.gif \
  --episodes 3

python scripts/generate_video.py \
  --model models/best_model.zip \
  --vecnorm models/lander_vecnorm.pkl \
  --env-id LunarLander-v3 \
  --output videos/lander_robust.gif \
  --episodes 3 \
  --robust

python scripts/generate_extreme_video.py \
  --model models/best_model.zip \
  --vecnorm models/lander_vecnorm.pkl \
  --env-id LunarLander-v3 \
  --output videos/lander_extreme.gif \
  --episodes 3

echo "Video generation complete!"
echo ""

# Step 3: Generate checksums for model files
echo "Step 3: Generating checksums..."
md5sum models/best_model_clean.zip models/best_model.zip models/lander_vecnorm.pkl > CHECKSUMS.txt

echo "Checksum generation complete!"
echo ""

# Step 4: Create release bundle
echo "Step 4: Creating release bundle..."
zip -r release_v1.1.zip \
  configs/ \
  scripts/ \
  models/best_model_clean.zip \
  models/best_model.zip \
  models/lander_vecnorm.pkl \
  videos/lander_clean.gif \
  videos/lander_robust.gif \
  videos/lander_extreme.gif \
  eval_results/receipt.json \
  RUNBOOK.md \
  CHECKSUMS.txt \
  MODEL_CARD.md \
  CHANGELOG.md

echo "Release bundle created: release_v1.1.zip"
echo ""

echo "=== Pipeline Complete! ==="
echo "All artifacts have been generated successfully."
echo "Check release_v1.1.zip for the final deliverables."