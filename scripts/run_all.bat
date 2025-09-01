@echo off
REM Comprehensive script to run the entire pipeline: fine-tune, evaluate, generate videos and artifacts

echo === Lunar Lander RL Pipeline ===
echo.

REM Step 1: Fine-tune the model with hybrid approach
echo Step 1: Fine-tuning model with hybrid approach (25%% clean, 75%% robust)...
python scripts/hybrid_finetune.py ^
  --model models/best_model.zip ^
  --vecnorm models/lander_vecnorm.pkl ^
  --output-dir models ^
  --seed 0

echo Fine-tuning complete!
echo.

REM Step 2: Generate videos for all environments
echo Step 2: Generating demonstration videos...
python scripts/generate_video.py ^
  --model models/best_model.zip ^
  --vecnorm models/lander_vecnorm.pkl ^
  --env-id LunarLander-v3 ^
  --output videos/lander_clean.gif ^
  --episodes 3

python scripts/generate_video.py ^
  --model models/best_model.zip ^
  --vecnorm models/lander_vecnorm.pkl ^
  --env-id LunarLander-v3 ^
  --output videos/lander_robust.gif ^
  --episodes 3 ^
  --robust

python scripts/generate_extreme_video.py ^
  --model models/best_model.zip ^
  --vecnorm models/lander_vecnorm.pkl ^
  --env-id LunarLander-v3 ^
  --output videos/lander_extreme.gif ^
  --episodes 3

echo Video generation complete!
echo.

REM Step 3: Generate checksums for model files
echo Step 3: Generating checksums...
powershell -Command "Get-FileHash -Algorithm MD5 models/best_model_clean.zip, models/best_model.zip, models/lander_vecnorm.pkl | ForEach-Object { $_.Path + ' ' + $_.Hash } | Out-File -FilePath CHECKSUMS.txt"

echo Checksum generation complete!
echo.

REM Step 4: Create release bundle
echo Step 4: Creating release bundle...
powershell -Command "Compress-Archive -Path configs/, scripts/, models/best_model_clean.zip, models/best_model.zip, models/lander_vecnorm.pkl, videos/lander_clean.gif, videos/lander_robust.gif, videos/lander_extreme.gif, eval_results/receipt.json, RUNBOOK.md, CHECKSUMS.txt, MODEL_CARD.md, CHANGELOG.md -DestinationPath release_v1.1.zip -Force"

echo Release bundle created: release_v1.1.zip
echo.

echo === Pipeline Complete! ===
echo All artifacts have been generated successfully.
echo Check release_v1.1.zip for the final deliverables.