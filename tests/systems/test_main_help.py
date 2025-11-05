import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata
]


def test_main_help(system, system_config):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "-h"
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)

    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
    assert "usage:" in result.stdout
