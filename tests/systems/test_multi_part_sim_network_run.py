import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata
]


def test_multi_part_sim_network_run(system, system_config, sim_output):
    if not system_config.get("multi-part-sim", False):
        pytest.skip(f"{system} does not support basic multi-part-sim run.")

    if not system_config.get("net", False):
        pytest.skip(f"{system} does not support network run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run-parts",
        "--time", "1h",
        "-x", f"{system}/*",
        "--net",
        "-o", sim_output,
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
