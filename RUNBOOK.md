# Operational Runbook: Autonomous Spacecraft Flight Guidance System

This runbook documents operational procedures for training, evaluation, sub-millisecond serving, latency benchmarking, and cryptographic verification.

---

## 1. Quick Reference Commands

| Task | Command | Target Latency / Metric |
| :--- | :--- | :--- |
| **Verify Integrity** | `bash scripts/verify.sh` or `lunar-lander verify` | 16/16 artifacts match SHA-256 |
| **Run Test Suite** | `pytest tests/ -v --cov=lunar_lander` or `make test` | 29/29 passed in <5 seconds |
| **Benchmark Latency** | `lunar-lander benchmark --iterations 2000` | P99 < 0.5 ms, QPS > 4,000 |
| **Launch Serving API** | `lunar-lander serve --host 0.0.0.0 --port 8000` | Health: `/healthz`, `/readyz` |
| **Run Clean Eval** | `lunar-lander eval --regime clean --episodes 100` | Expected reward: 263.5 ± 32 |
| **Run Robust Eval** | `lunar-lander eval --regime robust --episodes 100` | Expected reward: 219.6 ± 103 |
| **Run Extreme Eval** | `lunar-lander eval --regime extreme --episodes 100` | Active safety overrides: ~6% |

---

## 2. Environment Setup

### Using `uv` (Recommended)
```bash
# Create Python 3.12 virtualenv and install editable package with dev dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 3. Production Serving Procedures

### Starting the Guidance Daemon
```bash
# Launch FastAPI uvicorn daemon
export MODEL_PATH="models/best_model.zip"
export VECNORM_PATH="models/lander_vecnorm.pkl"
export DEVICE="cpu"

uvicorn lunar_lander.api:app --host 0.0.0.0 --port 8000 --workers 1
```

### Health Probing
```bash
# Liveness
curl -f http://localhost:8000/healthz

# Readiness (verifies model & normalization weights are loaded into memory)
curl -f http://localhost:8000/readyz

# Telemetry & Request Counters
curl http://localhost:8000/metrics
```

### Sending Single Guidance Query
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "x": 0.02,
    "y": 0.85,
    "vx": -0.01,
    "vy": -0.12,
    "angle": 0.01,
    "angular_velocity": 0.0,
    "leg1_contact": 0.0,
    "leg2_contact": 0.0
  }'
```

---

## 4. Empirical Evaluation Protocol

To detect potential regressions against the canonical baseline:

```bash
# Run 100-episode deterministic evaluation on all regimes and dump telemetry
lunar-lander eval --regime all --episodes 100 --output eval_results/telemetry.json
```

### Acceptance Tolerances
- **Clean Regime**: Mean reward must remain $\ge 240.0$. Solved landing rate must be $\ge 90\%$.
- **Robust Regime**: Mean reward must remain $\ge 200.0$. Crash rate must be $\le 5\%$.
- Any crash rate $> 10\%$ in robust conditions constitutes a performance regression.

---

## 5. Artifact Verification & Checksum Maintenance

Whenever retraining models or regenerating demonstration assets:

```bash
# Step 1: Recompute cryptographic hashes
python -c "
import hashlib, os
files = [
    'models/best_model.zip', 'models/best_model_clean.zip', 'models/lander_vecnorm.pkl',
    'videos/lander_clean.gif', 'videos/lander_robust.gif', 'videos/lander_extreme.gif',
    'eval_results/receipt.json', 'eval_results/telemetry.json',
    'assets/benchmark_telemetry.svg', 'assets/benchmark_telemetry.png',
    'assets/latency_profile.svg', 'assets/latency_profile.png'
]
lines = ['# SHA-256 Cryptographic Integrity Manifest', '# Format: <SHA-256>  <BYTE_SIZE>  <FILE_PATH>']
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        h = hashlib.sha256()
        with open(f, 'rb') as fp:
            while chunk := fp.read(65536): h.update(chunk)
        lines.append(f'{h.hexdigest()}  {size}  {f}')
with open('CHECKSUMS.txt', 'w') as out:
    out.write('\n'.join(lines) + '\n')
"

# Step 2: Validate manifest
lunar-lander verify
```