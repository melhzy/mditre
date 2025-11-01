# Makefile for MDITRE Development
# 
# Common development tasks automated for easy use
# Run 'make help' to see all available commands

.PHONY: help install install-dev test test-cov test-fast lint format clean docs

# Default target
help:
	@echo "MDITRE Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make test-fast    - Run tests (skip slow tests)"
	@echo "  make test-markers - Run tests by marker (e.g., make test-markers MARKER=architecture)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Check code style with flake8"
	@echo "  make format       - Format code with black and isort"
	@echo "  make typecheck    - Run mypy type checker"
	@echo "  make quality      - Run all code quality checks"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean        - Remove build artifacts and caches"
	@echo "  make clean-test   - Remove test artifacts"
	@echo "  make clean-all    - Remove all generated files"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         - Build documentation"
	@echo "  make docs-serve   - Serve documentation locally"
	@echo ""
	@echo "Package:"
	@echo "  make build        - Build distribution packages"
	@echo "  make check        - Verify package integrity"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=mditre --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	pytest tests/ -v -m "not slow"

test-markers:
	pytest tests/ -v -m "$(MARKER)"

test-parallel:
	pytest tests/ -v -n auto

# Code quality
lint:
	@echo "Running flake8..."
	flake8 mditre/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "✓ Lint checks passed"

format:
	@echo "Formatting with black..."
	black mditre/ tests/ --line-length=100
	@echo "Sorting imports with isort..."
	isort mditre/ tests/ --profile black
	@echo "✓ Code formatted"

typecheck:
	@echo "Running mypy type checker..."
	mypy mditre/ --ignore-missing-imports
	@echo "✓ Type checks passed"

quality: lint typecheck
	@echo "✓ All quality checks passed"

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	@echo "✓ Build artifacts cleaned"

clean-test:
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	@echo "✓ Test artifacts cleaned"

clean-all: clean clean-test
	@echo "Cleaning all generated files..."
	rm -rf logs/
	rm -rf mditre_outputs/
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ All artifacts cleaned"

# Documentation
docs:
	@echo "Building documentation..."
	@echo "TODO: Setup Sphinx documentation"
	@echo "Run: sphinx-build docs/source docs/build"

docs-serve:
	@echo "Serving documentation locally..."
	@echo "TODO: Setup documentation server"
	@echo "Run: python -m http.server --directory docs/build"

# Package building
build: clean
	@echo "Building distribution packages..."
	python -m build
	@echo "✓ Packages built in dist/"

check:
	@echo "Checking package integrity..."
	twine check dist/*
	@echo "✓ Package checks passed"

# Git hooks
hooks:
	@echo "Setting up pre-commit hooks..."
	@echo "TODO: Install pre-commit"
	@echo "Run: pre-commit install"

# Quick development cycle
dev: format test-fast
	@echo "✓ Development cycle complete"

# Full CI simulation
ci: quality test-cov
	@echo "✓ CI checks complete"

# Version bumping (placeholder)
version:
	@echo "Current version: 1.0.0"
	@echo "To bump version, edit:"
	@echo "  - setup.py"
	@echo "  - pyproject.toml"
	@echo "  - mditre/__init__.py"
	@echo "  - CHANGELOG.md"
