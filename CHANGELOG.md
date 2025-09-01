# Changelog

## v1.1 - Hybrid Fine-tuned Model

- Implemented hybrid fine-tuning with 25% clean and 75% robust environment mix
- Improved clean environment performance to 267.15 average reward
- Maintained robust environment performance at 235.58 average reward
- Used lower learning rate and adjusted entropy coefficient
- Applied early stopping when both performance gates were met

## v1.0 - Initial Robust Model

- Trained model with focus on robust environments
- Achieved strong performance in robust environments
- Clean environment performance regressed compared to baseline
- Established baseline for further improvements