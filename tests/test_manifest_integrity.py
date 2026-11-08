"""
Cryptographic integrity tests for repository checkpoints, configs, and assets.
"""

import os
import hashlib
import pytest


def test_manifest_file_exists():
    assert os.path.exists("CHECKSUMS.txt"), "CHECKSUMS.txt manifest must exist"


def test_all_manifest_checksums_match():
    with open("CHECKSUMS.txt", "r") as f:
        lines = f.readlines()

    tested_files = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        assert len(parts) >= 3, f"Invalid manifest line format: {line}"
        expected_sha256, expected_bytes, filepath = parts[0], int(parts[1]), parts[2]

        assert os.path.exists(filepath), f"File in manifest missing from disk: {filepath}"
        actual_bytes = os.path.getsize(filepath)
        assert actual_bytes == expected_bytes, f"Byte size mismatch for {filepath}: expected {expected_bytes}, got {actual_bytes}"

        h = hashlib.sha256()
        with open(filepath, "rb") as bf:
            while chunk := bf.read(65536):
                h.update(chunk)
        actual_sha256 = h.hexdigest()

        assert actual_sha256.lower() == expected_sha256.lower(), f"SHA-256 mismatch for {filepath}"
        tested_files += 1

    assert tested_files >= 10, f"Expected at least 10 tracked assets in manifest, tested {tested_files}"
