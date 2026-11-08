# Model Card: LunarLander PPO Autonomous Descent Guidance Policy

## Model Details
- **Developer**: Omer ([@Omerlol2004](https://github.com/Omerlol2004))
- **Model Architecture**: Multi-Layer Perceptron (MLP) Actor-Critic (`[64, 64]`, `tanh` activations)
- **Algorithm**: Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
- **Input Dimensions**: 8-dimensional continuous telemetry vector `[x, y, vx, vy, θ, ω, leg1_contact, leg2_contact]`
- **Output Dimensions**: 4 discrete thruster commands (`0: COAST`, `1: FIRE_LEFT`, `2: FIRE_MAIN`, `3: FIRE_RIGHT`)
- **Framework**: PyTorch 2.5+, Stable-Baselines3 v2.4+, Gymnasium v1.4+
- **Serving Layer**: FastAPI with sub-millisecond P99 latency (0.26 ms) on CPU
- **License**: MIT

---

## Intended Use & Safety Envelope
- **Primary Use**: Autonomous guidance, attitude control, and soft touchdown landing of a simulated planar lunar module under non-stationary aerodynamic turbulence.
- **Out-of-Scope / Non-Recommended Use**: Physical spacecraft flight control without high-TRL hardware-in-the-loop (HIL) validation.
- **Defensive Safety Guardrail**: A deterministic `SafetyFilter` wraps policy execution, overriding actions under severe attitude tumble ($|\omega| > 2.5$ rad/s), extreme tilt ($|\theta| > 0.45$ rad), or hazardous descent rate near ground ($y < 0.25$ m, $v_y < -1.2$ m/s).

---

## Training Data & Curriculum
- **Environment**: Gymnasium `LunarLander-v3`
- **Curriculum**:
  - **Stage A (Clean Baseline)**: 5.03M environment interactions on standard lunar gravity without atmospheric wind.
  - **Stage B (Robust Hybrid Fine-Tuning)**: 11.37M additional steps trained on a 25% clean / 75% turbulent aerodynamic mix (`wind_power = 15.0`, `turbulence_power = 1.5`). Total steps: 16.4M.
- **Observation Normalization**: Running empirical statistics (`lander_vecnorm.pkl`) frozen during inference with `norm_obs=True` and `norm_reward=False`.

---

## Quantitative Performance

Evaluation across 100 deterministic episodes per regime (seed=42):

| Operational Regime | Mean Reward | Median Reward | Interquartile Range | Solved Rate ($\ge 200$) | Safety Interventions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 263.46 ± 32.17 | 268.43 | 22.73 | 93.0% | 0.0% |
| **Robust (Wind 15.0)** | 219.60 ± 103.26 | 257.51 | 47.79 | 79.0% | 0.98% |
| **Extreme (Wind 25.0)** | -8.64 ± 150.19 | -35.80 | 142.84 | 17.0% | 6.09% |

### Latency Budget (Single-Core CPU)
- **P50 Latency**: 0.176 ms (176 µs)
- **P95 Latency**: 0.206 ms (206 µs)
- **P99 Latency**: 0.263 ms (263 µs)
- **Throughput**: 5,520 QPS

---

## Cryptographic Provenance
- `models/best_model.zip` (SHA-256): `72b190bca2840f3cfdaa9ab8e7caa63564c8cd88434810cbb53f97777248b652`
- `models/best_model_clean.zip` (SHA-256): `ee13d0b9b47f6de0b05b0138bb4c0b3673c9ad32121e1354d589f6f1419c6b89`
- `models/lander_vecnorm.pkl` (SHA-256): `55047fad2aac321cef03ab3387a37d847b02de88f6db2b1780b306c7d739d35f`