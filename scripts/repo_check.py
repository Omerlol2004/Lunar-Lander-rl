#!/usr/bin/env python3

import os
import sys
import glob
import json

def check_canonical_files():
    """Verify that all canonical files are present."""
    canonical_files = [
        "models/best_model.zip",
        "models/lander_vecnorm.pkl",
        "models/best_model_clean.zip",
        "eval_results/receipt.json"
    ]
    
    missing_files = []
    for file_path in canonical_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing canonical files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def check_videos_directory():
    """Verify only allowed files are in videos/ directory."""
    allowed_patterns = [
        "videos/lander_clean.gif",
        "videos/lander_robust.gif",
        "videos/lander_extreme.gif"
    ]
    
    # Get all files in videos directory
    all_files = glob.glob("videos/*")
    
    # Check for disallowed files
    disallowed_files = []
    for file_path in all_files:
        file_path = file_path.replace("\\", "/")  # Normalize path separators
        if file_path not in allowed_patterns:
            if "_new.gif" in file_path or "smoke_" in file_path or file_path.endswith(".mp4"):
                disallowed_files.append(file_path)
    
    if disallowed_files:
        print("ERROR: Disallowed files found in videos/ directory:")
        for file in disallowed_files:
            print(f"  - {file}")
        return False
    
    return True

def check_eval_results_directory():
    """Verify only receipt.json exists in eval_results/ directory."""
    json_files = glob.glob("eval_results/*.json")
    
    if len(json_files) > 1 or (len(json_files) == 1 and not os.path.exists("eval_results/receipt.json")):
        print("ERROR: Extra JSON files found in eval_results/ directory:")
        for file in json_files:
            if os.path.basename(file) != "receipt.json":
                print(f"  - {file}")
        return False
    
    return True

def check_archive_directory():
    """Verify no files in archive/ directory are tracked by Git."""
    import subprocess
    
    # Run git ls-files to check for tracked files in archive/
    result = subprocess.run(["git", "ls-files", "archive/"], capture_output=True, text=True)
    tracked_files = result.stdout.strip()
    
    if tracked_files:
        print("ERROR: Files in archive/ directory are tracked by Git:")
        for file in tracked_files.split('\n'):
            print(f"  - {file}")
        print("The archive/ directory should remain untracked.")
        return False
    
    return True

def main():
    """Run all repository checks."""
    checks = [
        (check_canonical_files, "Canonical files check"),
        (check_videos_directory, "Videos directory check"),
        (check_eval_results_directory, "Eval results directory check"),
        (check_archive_directory, "Archive directory check")
    ]
    
    all_passed = True
    
    for check_func, check_name in checks:
        print(f"Running {check_name}...")
        if not check_func():
            all_passed = False
            print(f"{check_name} failed!\n")
        else:
            print(f"{check_name} passed!\n")
    
    if all_passed:
        print("OK: All repository checks passed!")
        return 0
    else:
        print("FAILED: Repository checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())