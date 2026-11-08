#!/usr/bin/env python3
"""
Repository sanity and integrity checker.
Validates presence of canonical checkpoints, evaluation receipts, and clean directories.
"""

import os
import sys
import glob
import subprocess


def check_canonical_files():
    """Verify that all canonical files are present."""
    canonical_files = [
        "models/best_model.zip",
        "models/lander_vecnorm.pkl",
        "models/best_model_clean.zip",
        "eval_results/receipt.json",
        "CHECKSUMS.txt",
    ]
    missing = [f for f in canonical_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing canonical files:")
        for f in missing:
            print(f"  - {f}")
        return False
    return True


def check_videos_directory():
    """Verify canonical video demonstration files exist."""
    required = [
        "videos/lander_clean.gif",
        "videos/lander_robust.gif",
        "videos/lander_extreme.gif",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"WARNING: Video demonstration files missing: {missing}")
        return False
    return True


def check_eval_results_directory():
    """Verify only allowed JSON receipt files exist in eval_results/."""
    json_files = glob.glob("eval_results/*.json")
    allowed = {"receipt.json", "telemetry.json"}
    for f in json_files:
        basename = os.path.basename(f)
        if basename not in allowed:
            print(f"ERROR: Unexpected artifact in eval_results/: {f}")
            return False
    return True


def check_archive_directory():
    """Verify no files in archive/ directory are tracked by Git."""
    result = subprocess.run(["git", "ls-files", "archive/"], capture_output=True, text=True)
    tracked = result.stdout.strip()
    if tracked:
        print("ERROR: Tracked files found in archive/ directory:")
        print(tracked)
        return False
    return True


def main():
    checks = [
        (check_canonical_files, "Canonical files check"),
        (check_videos_directory, "Videos directory check"),
        (check_eval_results_directory, "Eval results directory check"),
        (check_archive_directory, "Archive directory check"),
    ]
    all_passed = True
    for func, name in checks:
        print(f"Running {name}...")
        if not func():
            all_passed = False
            print(f"FAILED: {name}\n")
        else:
            print(f"PASSED: {name}\n")

    if all_passed:
        print("OK: All repository checks passed!")
        return 0
    else:
        print("FAILED: Repository sanity checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())