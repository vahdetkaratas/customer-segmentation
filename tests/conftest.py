"""
Pytest configuration and fixtures for customer segmentation tests
"""

import pytest
from pathlib import Path

@pytest.fixture
def project_root():
    """Return the project root directory as a Path object."""
    return Path(__file__).parent.parent

@pytest.fixture
def data_dir(project_root):
    """Return the data directory as a Path object."""
    return project_root / "data"

@pytest.fixture
def processed_data_dir(data_dir):
    """Return the processed data directory as a Path object."""
    return processed_data_dir / "processed"

@pytest.fixture
def models_dir(project_root):
    """Return the models directory as a Path object."""
    return project_root / "models"

@pytest.fixture
def reports_dir(project_root):
    """Return the reports directory as a Path object."""
    return project_root / "reports"
