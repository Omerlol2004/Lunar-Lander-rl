# Autonomous Spacecraft Descent Policy - Enterprise Makefile

.PHONY: verify clean test serve benchmark eval lint help

help:
	@echo "Available commands:"
	@echo "  make test       - Run hermetic automated test suite with coverage"
	@echo "  make benchmark  - Run sub-millisecond policy latency benchmark"
	@echo "  make eval       - Run empirical evaluation across Clean, Robust, and Extreme regimes"
	@echo "  make serve      - Launch FastAPI serving daemon on port 8000"
	@echo "  make verify     - Verify SHA-256 cryptographic manifests and repo integrity"
	@echo "  make clean      - Clean temporary files and build artifacts"

verify:
	python -m lunar_lander.cli verify --checksum-file CHECKSUMS.txt
	python scripts/repo_check.py

test:
	pytest tests/ -v --cov=lunar_lander --cov-report=term-missing

benchmark:
	python -m lunar_lander.cli benchmark --iterations 2000

eval:
	python -m lunar_lander.cli eval --regime all --episodes 100

serve:
	python -m lunar_lander.cli serve --host 0.0.0.0 --port 8000

clean:
	@echo "Cleaning temporary files..."
	find . -name "*.tmp" -delete
	find . -name "*.bak" -delete
	find . -name "*.log" -delete
	find . -name ".DS_Store" -delete
	find . -name "Thumbs.db" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info