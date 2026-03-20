import pytest
import os
import importlib.util
from pathlib import Path
from typing import Generator, List


def get_example_files() -> List[Path]:
    """Get all Python files from the examples directory."""
    package_root = Path(__file__).parent.parent
    examples_dir = package_root / "examples"
    return list(examples_dir.glob("**/*.py"))


def import_module_from_path(path: Path) -> object:
    """Dynamically import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_files() -> Generator[List[Path], None, None]:
    """Fixture that provides all example files."""
    files = get_example_files()
    if not files:
        pytest.skip("No example files found")
    yield files


@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda x: x.name)
def test_example(example_file):
    """Test that each example file can be imported."""
    try:
        import_module_from_path(example_file)
    except Exception as e:
        pytest.fail(f"Failed to import {example_file}: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
