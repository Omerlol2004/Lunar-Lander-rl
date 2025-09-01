#!/bin/bash

# Repository verification script
# Run this to verify the repository is in a consistent state

echo "Running repository verification checks..."
python scripts/repo_check.py

exit $?