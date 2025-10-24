"""
Simple notebook validation tests for ODH Data Processing.
"""

import os
from pathlib import Path

import pytest
from notebook_validators import NotebookValidator


def pytest_generate_tests(metafunc):
    """Generate tests dynamically based on discovered or changed notebooks."""
    if "notebook_path" in metafunc.fixturenames:
        # Check if we have changed notebooks from CI environment
        changed_notebooks_env = os.environ.get("CHANGED_NOTEBOOKS")

        if changed_notebooks_env:
            # Parse changed notebooks from CI
            import json

            try:
                changed_files = json.loads(changed_notebooks_env)
                notebook_paths = [
                    Path(file) for file in changed_files if file.endswith(".ipynb")
                ]
                existing_notebooks = [path for path in notebook_paths if path.exists()]

                if existing_notebooks:
                    metafunc.parametrize(
                        "notebook_path", existing_notebooks, ids=lambda x: x.name
                    )
                    return
            except (json.JSONDecodeError, Exception):
                pass

        # Fallback: discover all notebooks
        notebooks_dir = Path("notebooks")
        if notebooks_dir.exists():
            notebook_files = list(notebooks_dir.rglob("*.ipynb"))
            if notebook_files:
                metafunc.parametrize(
                    "notebook_path", notebook_files, ids=lambda x: x.name
                )
            else:
                metafunc.parametrize("notebook_path", [], ids=[])
        else:
            metafunc.parametrize("notebook_path", [], ids=[])


def test_notebook_format_valid(notebook_path):
    """Test that notebook has valid format."""
    if not notebook_path:
        pytest.skip("No notebooks to validate")

    validator = NotebookValidator(notebook_path)
    validator.load_notebook()
    assert validator.notebook is not None, f"Failed to load notebook: {notebook_path}"


def test_notebook_clean_state(notebook_path):
    """Test that notebook has clean execution state."""
    if not notebook_path:
        pytest.skip("No notebooks to validate")

    validator = NotebookValidator(notebook_path)
    validator.load_notebook()

    output_errors = validator.validate_no_outputs()
    assert not output_errors, "Notebook has cell outputs:\n" + "\n".join(output_errors)

    count_errors = validator.validate_no_execution_counts()
    assert not count_errors, "Notebook has execution counts:\n" + "\n".join(
        count_errors
    )


def test_notebook_has_parameters(notebook_path):
    """Test that notebook has parameters cell."""
    if not notebook_path:
        pytest.skip("No notebooks to validate")

    validator = NotebookValidator(notebook_path)
    validator.load_notebook()

    param_errors = validator.validate_parameters_cell_exists()
    assert not param_errors, "Notebook missing parameters cell:\n" + "\n".join(
        param_errors
    )
