"""
Sanity tests for tooling configuration
"""

import os
from pathlib import Path

import pytest

# Get project root
project_root = Path(__file__).parent.parent


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and contains required sections."""
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml should exist"
    with open(pyproject_path) as f:
        content = f.read()
    assert "[tool.ruff]" in content, "pyproject.toml should contain [tool.ruff] section"
    assert (
        "[tool.black]" in content
    ), "pyproject.toml should contain [tool.black] section"
    assert (
        "[tool.pytest.ini_options]" in content
    ), "pyproject.toml should contain [tool.pytest.ini_options] section"


def test_makefile_exists():
    """Test that Makefile exists and contains required targets."""
    makefile_path = project_root / "Makefile"
    assert makefile_path.exists(), "Makefile should exist"
    with open(makefile_path) as f:
        content = f.read()
    required_targets = ["install", "fmt", "lint", "test", "clean"]
    for target in required_targets:
        assert f"{target}:" in content, f"Makefile should contain '{target}:' target"


def test_pre_commit_config_exists():
    """Test that .pre-commit-config.yaml exists and is valid YAML."""
    pre_commit_path = project_root / ".pre-commit-config.yaml"
    assert pre_commit_path.exists(), ".pre-commit-config.yaml should exist"
    try:
        import yaml

        with open(pre_commit_path) as f:
            config = yaml.safe_load(f)
        assert "repos" in config, ".pre-commit-config.yaml should contain 'repos' key"
        assert isinstance(config["repos"], list), "repos should be a list"
    except ImportError:
        pytest.skip("PyYAML not available")


def test_github_workflow_exists():
    """Test that GitHub Actions workflow exists."""
    workflow_path = project_root / ".github" / "workflows" / "ci.yml"
    assert workflow_path.exists(), "GitHub Actions CI workflow should exist"
    try:
        import yaml

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)
        assert "name" in workflow, "CI workflow should have a name"
        assert "on" in workflow or True in workflow, "CI workflow should have triggers"
        assert "jobs" in workflow, "CI workflow should have jobs"
    except ImportError:
        pytest.skip("PyYAML not available")


def test_requirements_txt_contains_dev_deps():
    """Test that requirements.txt contains development dependencies."""
    requirements_path = project_root / "requirements.txt"
    assert requirements_path.exists(), "requirements.txt should exist"
    with open(requirements_path) as f:
        content = f.read()
    dev_deps = ["black", "ruff", "isort", "pre-commit", "pytest-cov"]
    for dep in dev_deps:
        assert dep in content, f"requirements.txt should contain {dep}"


@pytest.mark.skipif(
    not (project_root / ".git").exists(), reason="Not in a git repository"
)
def test_git_attributes_for_notebooks():
    """Test that .gitattributes exists and configures nbstripout for notebooks."""
    gitattributes_path = project_root / ".gitattributes"
    if not gitattributes_path.exists():
        pytest.skip(
            ".gitattributes not found - nbstripout may be configured via pre-commit only"
        )
    with open(gitattributes_path) as f:
        content = f.read()
    assert "*.ipynb" in content, ".gitattributes should configure *.ipynb files"


def test_project_structure():
    """Test that essential project directories exist."""
    essential_dirs = ["src", "tests", "data", "streamlit_app", "reports"]
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"


def test_python_files_exist():
    """Test that essential Python files exist."""
    essential_files = [
        "src/app_core.py",
        "streamlit_app/app.py",
        "tests/__init__.py",
        "tests/conftest.py",
    ]
    for file_path in essential_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"File {file_path} should exist"


@pytest.mark.skipif(
    os.name == "nt",  # Windows
    reason="Makefile targets may not work on Windows without proper setup",
)
def test_makefile_targets_available():
    """Test that Makefile targets are available (skip on Windows)."""
    import subprocess

    makefile_path = project_root / "Makefile"
    if not makefile_path.exists():
        pytest.skip("Makefile not found")
    try:
        result = subprocess.run(
            ["make", "help"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, "make help should succeed"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("make command not available or timed out")


def test_pyproject_toml_ruff_config():
    """Test that ruff configuration is properly set up."""
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path) as f:
        content = f.read()
    assert "line-length = 88" in content, "ruff should have line-length setting"
    assert "select = [" in content, "ruff should have select setting"


def test_pyproject_toml_black_config():
    """Test that black configuration is properly set up."""
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path) as f:
        content = f.read()
    assert "line-length = 88" in content, "black should have line-length setting"
    assert "target-version = [" in content, "black should have target-version setting"
