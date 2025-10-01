# PlanCritic Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-fast test-gpu test-integration lint format clean build docs serve-docs web-dev web-build deploy-web docker-build docker-run benchmark profile

# Default target
help:
	@echo "PlanCritic Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install package in development mode with dev dependencies"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests with coverage"
	@echo "  test-fast        Run fast tests only (exclude slow and GPU tests)"
	@echo "  test-gpu         Run GPU-specific tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-unit        Run unit tests only"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run linting checks (flake8, mypy, black --check)"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Web Viewer Commands:"
	@echo "  web-dev          Start web viewer development server"
	@echo "  web-build        Build web viewer for production"
	@echo "  web-test         Run web viewer tests"
	@echo ""
	@echo "Build and Deploy Commands:"
	@echo "  build            Build package for distribution"
	@echo "  clean            Clean build artifacts and cache files"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Performance Commands:"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  profile          Run profiling analysis"
	@echo ""
	@echo "Utility Commands:"
	@echo "  check-deps       Check for outdated dependencies"
	@echo "  security-check   Run security vulnerability checks"

# Setup Commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,docs,web]"
	pre-commit install

# Testing Commands
test:
	pytest tests/ \
		--cov=src/plancritic \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--durations=10 \
		-v

test-fast:
	pytest tests/ \
		-m "not slow and not gpu and not integration" \
		--cov=src/plancritic \
		--cov-report=term-missing \
		-v

test-gpu:
	pytest tests/ \
		-m "gpu" \
		--cov=src/plancritic \
		--cov-report=term-missing \
		-v

test-integration:
	pytest tests/ \
		-m "integration" \
		--cov=src/plancritic \
		--cov-report=term-missing \
		-v

test-unit:
	pytest tests/ \
		-m "unit" \
		--cov=src/plancritic \
		--cov-report=term-missing \
		-v

test-models:
	pytest tests/test_models.py -v

test-physics:
	pytest tests/test_physics.py -v

test-data:
	pytest tests/test_data.py -v

test-cli:
	pytest tests/test_cli.py -v

# Code Quality Commands
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Running black check..."
	black --check src/ tests/
	@echo "Running isort check..."
	isort --check-only src/ tests/
	@echo "Running mypy..."
	mypy src/plancritic/

format:
	@echo "Formatting with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/
	@echo "Code formatting complete!"

type-check:
	mypy src/plancritic/ --strict

# Documentation Commands
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

docs-clean:
	cd docs && make clean

# Web Viewer Commands
web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

web-test:
	cd web && npm test

web-lint:
	cd web && npm run lint

web-install:
	cd web && npm install

# Build and Deploy Commands
build:
	python -m build

build-wheel:
	python setup.py bdist_wheel

build-sdist:
	python setup.py sdist

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

# Docker Commands
docker-build:
	docker build -t plancritic:latest .

docker-run:
	docker run -it --rm \
		--gpus all \
		-p 8888:8888 \
		-p 3000:3000 \
		-v $(PWD):/workspace \
		plancritic:latest

docker-run-cpu:
	docker run -it --rm \
		-p 8888:8888 \
		-p 3000:3000 \
		-v $(PWD):/workspace \
		plancritic:latest

# Performance Commands
benchmark:
	python -m pytest benchmarks/ -v --benchmark-only

profile:
	python -m cProfile -o profile_output.prof scripts/profile_analysis.py
	python -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(20)"

# Utility Commands
check-deps:
	pip list --outdated

security-check:
	safety check
	bandit -r src/

pre-commit-all:
	pre-commit run --all-files

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test-fast' to verify installation"

quick-check: format lint test-fast
	@echo "Quick development check complete!"

full-check: format lint test
	@echo "Full development check complete!"

# CI/CD simulation
ci-test:
	@echo "Running CI test suite..."
	make lint
	make test
	make build
	@echo "CI test suite complete!"

# Release preparation
prepare-release: clean full-check build docs
	@echo "Release preparation complete!"
	@echo "Ready for deployment"

# Jupyter notebook management
notebook-clean:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

notebook-test:
	pytest --nbval notebooks/

# Data management
download-sample-data:
	mkdir -p data/samples
	# Add commands to download sample datasets
	@echo "Sample data download complete!"

# Environment management
create-env:
	conda create -n plancritic python=3.9 -y
	@echo "Conda environment 'plancritic' created"
	@echo "Activate with: conda activate plancritic"

# Monitoring and logging
logs:
	tail -f logs/plancritic.log

# Database management (if applicable)
db-migrate:
	# Add database migration commands if needed
	@echo "Database migrations complete"

# Backup and restore
backup-models:
	mkdir -p backups/models
	cp -r models/ backups/models/$(shell date +%Y%m%d_%H%M%S)/
	@echo "Models backed up"

# Performance monitoring
monitor-gpu:
	watch -n 1 nvidia-smi

monitor-cpu:
	htop

# Help for specific components
help-web:
	@echo "Web Viewer Commands:"
	@echo "  web-install      Install web dependencies"
	@echo "  web-dev          Start development server"
	@echo "  web-build        Build for production"
	@echo "  web-test         Run web tests"
	@echo "  web-lint         Lint web code"

help-docker:
	@echo "Docker Commands:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run with GPU support"
	@echo "  docker-run-cpu   Run CPU-only version"

help-test:
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-fast        Quick tests only"
	@echo "  test-gpu         GPU tests"
	@echo "  test-integration Integration tests"
	@echo "  test-unit        Unit tests only"
	@echo "  test-models      Model tests"
	@echo "  test-physics     Physics tests"
	@echo "  test-data        Data processing tests"
	@echo "  test-cli         CLI tests"