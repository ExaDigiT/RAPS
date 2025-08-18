import sys
import tomllib
from pathlib import Path


def check_python_version():
    # Load pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    # Extract required python version (e.g., ">=3.11")
    requires_python = pyproject_data["project"]["requires-python"]

    # Get the minimum major/minor from the string
    # This assumes format like ">=3.11"
    version_str = requires_python.lstrip(">=").strip()
    required_major, required_minor = map(int, version_str.split(".")[:2])

    # Compare
    if sys.version_info < (required_major, required_minor):
        sys.stderr.write(
            f"Error: RAPS requires Python {required_major}.{required_minor} or greater\n"
        )
        sys.exit(1)
