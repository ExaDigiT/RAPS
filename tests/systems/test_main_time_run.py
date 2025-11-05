import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.time
]


@pytest.mark.parametrize("time_args", [
    "0", "1", "3600", "7200",
    pytest.param("43200", marks=pytest.mark.long),  # mark this one as long
    "0s", "1s", "3600s", "7200s",
    pytest.param("43200s", marks=pytest.mark.long),  # mark this one as long
    "0m", "1m", "60m",
    "0h", "1h",
    pytest.param("6h", marks=pytest.mark.long),  # mark this one as long
])
def test_main_time_run(system, system_config, time_args, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", time_args,
        "--system", system,
        "--noui",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
