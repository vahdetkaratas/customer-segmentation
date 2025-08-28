.PHONY: help install fmt lint test test-cov app clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install    - Create virtual environment and install dependencies"
	@echo "  fmt        - Format code with black, ruff, and isort"
	@echo "  lint       - Run linting checks"
	@echo "  test       - Run tests"
	@echo "  test-cov   - Run tests with coverage"
	@echo "  app        - Run Streamlit app"
	@echo "  clean      - Remove cache and build artifacts"

# Create virtual environment and install dependencies
install:
	@echo "Setting up development environment..."
	@if not exist ".venv" ( \
		echo "Creating virtual environment..." && \
		python -m venv .venv \
	)
	@echo "Activating virtual environment and installing dependencies..."
	@.venv\Scripts\activate && python -m pip install --upgrade pip
	@.venv\Scripts\activate && pip install -r requirements.txt
	@.venv\Scripts\activate && pre-commit install
	@echo "Development environment setup complete!"

# Format code
fmt:
	@echo "Formatting code..."
	@.venv\Scripts\activate && black .
	@.venv\Scripts\activate && ruff --fix .
	@.venv\Scripts\activate && isort .
	@echo "Code formatting complete!"

# Run linting checks
lint:
	@echo "Running linting checks..."
	@.venv\Scripts\activate && ruff .
	@.venv\Scripts\activate && isort --check-only .
	@.venv\Scripts\activate && black --check .
	@echo "Linting complete!"

# Run tests
test:
	@echo "Running tests..."
	@.venv\Scripts\activate && pytest -q

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	@.venv\Scripts\activate && pytest --cov=src --cov-report=term-missing

# Run Streamlit app
app:
	@echo "Starting Streamlit app..."
	@.venv\Scripts\activate && streamlit run streamlit_app\app.py

# Clean cache and build artifacts
clean:
	@echo "Cleaning cache and build artifacts..."
	@if exist "__pycache__" rmdir /s /q __pycache__
	@if exist ".pytest_cache" rmdir /s /q .pytest_cache
	@if exist ".ruff_cache" rmdir /s /q .ruff_cache
	@if exist ".coverage" del .coverage
	@if exist "coverage.xml" del coverage.xml
	@if exist "htmlcov" rmdir /s /q htmlcov
	@if exist "build" rmdir /s /q build
	@if exist "dist" rmdir /s /q dist
	@if exist "*.egg-info" rmdir /s /q *.egg-info
	@echo "Cleanup complete!"

# Quick check: format, lint, and test
check: fmt lint test
	@echo "All checks passed!"

# Install and run all checks
setup: install check
	@echo "Setup and checks complete!"
