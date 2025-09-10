import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.network
]


def test_main_network_run(system, system_config, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main run.")

    if not system_config.get("net", False):
        pytest.skip(f"{system} does not support network run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", "1m",
        "--system", system,
        "--net",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
