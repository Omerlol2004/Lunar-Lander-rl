# Reinforcement Learning Project Makefile

.PHONY: verify clean

verify:
	python scripts/repo_check.py

clean:
	@echo "Cleaning temporary files..."
	find . -name "*.tmp" -delete
	find . -name "*.bak" -delete
	find . -name "*.log" -delete
	find . -name ".DS_Store" -delete
	find . -name "Thumbs.db" -delete