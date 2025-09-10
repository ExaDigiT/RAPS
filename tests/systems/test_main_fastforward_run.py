import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.fastforward
]


@pytest.mark.parametrize("ff_arg", [
    "0", "1", "3600", "7200", "43200",
    "0s", "1s", "3600s", "7200s", "43200s",
    "0m", "1m", "60m",
    "0h", "1h", "6h",
])
def test_main_fastforward_run(system, system_config, ff_arg, sim_output):
    if not system_config.get("fastforward", False):
        pytest.skip(f"{system} does not support basic main run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "-t 1",
        "--fastforward", ff_arg,
        "--system", system,
        "--noui",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
