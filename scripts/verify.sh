#!/usr/bin/env bash
set -e

# Detect python executable (prefer active virtualenv or local .venv)
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "Using Python: $PYTHON_BIN"

echo "=== Step 1: Cryptographic Manifest Verification (SHA-256) ==="
"$PYTHON_BIN" -m lunar_lander.cli verify --checksum-file CHECKSUMS.txt

echo -e "\n=== Step 2: Canonical File and Repository Structure Check ==="
"$PYTHON_BIN" scripts/repo_check.py

echo -e "\nAll verification checks passed successfully!"