
import subprocess
import os
from pathlib import Path

import pytest
pytestmark = pytest.mark.nodata

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adjust if needed


@pytest.mark.order(1)
def test_main_withui():
    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
                            "python", "main.py", "run",
                            "--time", "1h",
                            ], capture_output=True,
                            text=True
                            )
    assert result.returncode == 0


@pytest.mark.order(2)
def test_main_noui():
    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
                            "python", "main.py", "run",
                            "--time", "1h",
                            "--noui"
                            ], capture_output=True,
                            text=True
                            )
    assert result.returncode == 0


@pytest.mark.long
@pytest.mark.order(3)
def test_main_long():
    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
                            "python", "main.py", "run",
                            ], capture_output=True,
                            text=True
                            )
    assert result.returncode == 0
