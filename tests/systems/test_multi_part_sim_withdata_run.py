import os
import subprocess
import gc
import pytest
from tests.util import PROJECT_ROOT, DATA_PATH


pytestmark = [
    pytest.mark.system,
    pytest.mark.withdata,
    pytest.mark.long
]


def test_multi_part_sim_withdata_run(system, system_config, system_files):
    if not system_config.get("multi-part-sim", False):
        pytest.skip(f"{system} does not support basic multi-part-sim run even without data.")
    if not system_config.get("withdata", False):
        pytest.skip(f"{system} does not support multi-part-sim run with data.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run-parts",
        "--time", "1h",
        "-x", f"{system}/*",
        "-f", ','.join(system_files),
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
