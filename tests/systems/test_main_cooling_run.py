import os
import subprocess
import gc
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.cooling,
]


def test_main_run(system, system_config):
    if not system_config.get("cooling", False):
        pytest.skip(f"{system} does not support cooling.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py",
        "--time", "1h",
        "--system", system,
        "-c",
        "--noui"
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
    del result
    gc.collect()
