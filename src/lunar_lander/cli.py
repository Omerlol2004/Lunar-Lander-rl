"""
Unified Enterprise Command-Line Interface for Lunar Lander Autonomous Flight.
Commands:
  - eval: Run empirical evaluation across Clean, Robust, or Extreme regimes.
  - serve: Start the sub-millisecond FastAPI serving daemon.
  - benchmark: Execute microsecond latency profiling (P50, P95, P99, QPS).
  - verify: Verify SHA-256 checksums and integrity manifests.
"""

import argparse
import sys
import os
import json
import time
import hashlib
import numpy as np

from lunar_lander.policy import PolicyEngine
from lunar_lander.evaluator import Evaluator
from lunar_lander.config import SafetyConfig


def cmd_eval(args):
    """Execute evaluation protocol."""
    print(f"Loading policy engine from '{args.model}'...")
    engine = PolicyEngine(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        device=args.device,
    )
    evaluator = Evaluator(engine)

    regimes = ["clean", "robust", "extreme"] if args.regime == "all" else [args.regime]
    results = {}

    for reg in regimes:
        print(f"\n--- Running {reg.upper()} Evaluation ({args.episodes} episodes, seed={args.seed}) ---")
        metrics = evaluator.evaluate(
            regime=reg,
            episodes=args.episodes,
            seed=args.seed,
            apply_safety=not args.disable_safety,
        )
        results[reg] = metrics.model_dump()
        print(json.dumps(metrics.model_dump(), indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved evaluation metrics to '{args.output}'")


def cmd_benchmark(args):
    """Profile inference latency and compute throughput."""
    print(f"Initializing PolicyEngine for latency profiling ({args.iterations} iterations)...")
    engine = PolicyEngine(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        device=args.device,
        warmup=True,
    )

    dummy_state = np.array([0.05, 0.4, 0.0, -0.15, 0.02, 0.01, 0.0, 0.0], dtype=np.float32)
    latencies = []

    for _ in range(args.iterations):
        t0 = time.perf_counter()
        engine.predict(dummy_state, deterministic=True, apply_safety=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    qps = 1000.0 / np.mean(arr)

    report = {
        "iterations": args.iterations,
        "mean_latency_ms": round(float(np.mean(arr)), 3),
        "std_latency_ms": round(float(np.std(arr)), 3),
        "p50_latency_ms": round(float(p50), 3),
        "p95_latency_ms": round(float(p95), 3),
        "p99_latency_ms": round(float(p99), 3),
        "throughput_qps": round(float(qps), 1),
    }

    print("\n================ LATENCY BENCHMARK ================")
    print(json.dumps(report, indent=2))
    print("===================================================\n")


def cmd_verify(args):
    """Verify cryptographic SHA-256 and byte sizes from CHECKSUMS.txt."""
    checksum_file = args.checksum_file
    if not os.path.exists(checksum_file):
        print(f"ERROR: Checksums manifest not found: {checksum_file}")
        sys.exit(1)

    print(f"Verifying repository artifacts against '{checksum_file}'...\n")
    all_ok = True
    verified_count = 0

    with open(checksum_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            expected_sha256, expected_bytes, filepath = parts[0], int(parts[1]), parts[2]
            if not os.path.exists(filepath):
                print(f"FAIL [MISSING]: {filepath}")
                all_ok = False
                continue

            actual_bytes = os.path.getsize(filepath)
            h = hashlib.sha256()
            with open(filepath, "rb") as bf:
                while chunk := bf.read(65536):
                    h.update(chunk)
            actual_sha256 = h.hexdigest()

            if actual_bytes != expected_bytes:
                print(f"FAIL [SIZE MISMATCH]: {filepath} (expected {expected_bytes}, got {actual_bytes})")
                all_ok = False
            elif actual_sha256.lower() != expected_sha256.lower():
                print(f"FAIL [SHA256 MISMATCH]: {filepath}")
                all_ok = False
            else:
                print(f"PASS: {filepath} ({actual_bytes} bytes, SHA256 verified)")
                verified_count += 1

    if all_ok and verified_count > 0:
        print(f"\nSUCCESS: All {verified_count} artifacts cryptographically verified.")
        sys.exit(0)
    else:
        print(f"\nFAILURE: Verification failed.")
        sys.exit(1)


def cmd_serve(args):
    """Start uvicorn production server."""
    import uvicorn
    os.environ["MODEL_PATH"] = args.model
    if args.vecnorm:
        os.environ["VECNORM_PATH"] = args.vecnorm
    os.environ["DEVICE"] = args.device

    print(f"Starting Lunar Lander Flight Guidance API on {args.host}:{args.port}...")
    uvicorn.run("lunar_lander.api:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        prog="lunar-lander",
        description="Production CLI for Lunar Lander Autonomous Guidance System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: eval
    p_eval = subparsers.add_parser("eval", help="Run empirical evaluation protocol")
    p_eval.add_argument("--regime", choices=["clean", "robust", "extreme", "all"], default="clean")
    p_eval.add_argument("--episodes", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--model", type=str, default="models/best_model.zip")
    p_eval.add_argument("--vecnorm", type=str, default="models/lander_vecnorm.pkl")
    p_eval.add_argument("--device", type=str, default="cpu")
    p_eval.add_argument("--disable-safety", action="store_true")
    p_eval.add_argument("--output", type=str, default=None)
    p_eval.set_defaults(func=cmd_eval)

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Profile inference latency and throughput")
    p_bench.add_argument("--iterations", type=int, default=2000)
    p_bench.add_argument("--model", type=str, default="models/best_model.zip")
    p_bench.add_argument("--vecnorm", type=str, default="models/lander_vecnorm.pkl")
    p_bench.add_argument("--device", type=str, default="cpu")
    p_bench.set_defaults(func=cmd_benchmark)

    # Subcommand: verify
    p_verify = subparsers.add_parser("verify", help="Verify cryptographic SHA-256 checksums")
    p_verify.add_argument("--checksum-file", type=str, default="CHECKSUMS.txt")
    p_verify.set_defaults(func=cmd_verify)

    # Subcommand: serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI serving microservice")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", type=str, default="models/best_model.zip")
    p_serve.add_argument("--vecnorm", type=str, default="models/lander_vecnorm.pkl")
    p_serve.add_argument("--device", type=str, default="cpu")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
