import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT, DATA_PATH


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.withdata,
    pytest.mark.long,
    pytest.mark.network
]


def test_main_network_withdata_run(system, system_config, system_files, sim_output):
    if not system_config.get("net", False):
        pytest.skip(f"{system} does not support basic net run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", "1m",
        "--system", system,
        "-f", ','.join(system_files),
        "--net",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
