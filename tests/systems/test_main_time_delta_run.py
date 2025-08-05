import os
import subprocess
import gc
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.time_delta
]


@pytest.mark.parametrize("time_arg, tdelta_arg", [
    ("100", "1"),
    ("100", "1s"),
    ("100", "10s"),
    ("10m", "1m"),
    ("10h", "1h"),
    ("10h", "3h"),
    ("3d", "1d")
], ids=["1","1s","10s","1m","1h","3h","1d"])
def test_main_time_delta_run(system, system_config, time_arg, tdelta_arg):
    if not system_config.get("time_delta", False):
        pytest.skip(f"{system} does not support time_delta run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py",
        "-t", time_arg,
        "--time-delta", tdelta_arg,
        "--system", system,
        #--"-f", system_file,
        "--noui"
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
    del result
    gc.collect()
