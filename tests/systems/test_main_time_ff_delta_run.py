import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.time_delta
]


@pytest.mark.parametrize("time_arg, tdelta_arg, ff_arg", [
    ("100", "1", "103"),
    ("100", "1s", "2s"),
    ("100", "10s", "10s"),
    ("10m", "1m", "1m"),
    ("10h", "1h", "2h"),
    ("10h", "3h", "1h"),
    pytest.param("3d", "1d", "1d", marks=pytest.mark.long, id="1d (long)"),
], ids=["1", "1s", "10s", "1m", "1h", "3h", "1d"])
def test_main_time_ff_delta_run(system, system_config, time_arg, tdelta_arg,
                                ff_arg, sim_output):
    if not system_config.get("time_delta", False):
        pytest.skip(f"{system} does not support time_delta run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "-t", time_arg,
        "--ff", ff_arg,
        "--time-delta", tdelta_arg,
        "--system", system,
        "--noui",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
