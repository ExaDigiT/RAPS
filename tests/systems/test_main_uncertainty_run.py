import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.uncertainty,
    pytest.mark.long
]


def test_main_uncertainty_run(system, system_config, sim_output):
    if not system_config.get("uncertainty", False):
        pytest.skip(f"{system} does not support uncertainty.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", "3m",
        "--system", system,
        "-u",
        "--noui",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
