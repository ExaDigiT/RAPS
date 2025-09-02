import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT, DATA_PATH


pytestmark = [
    pytest.mark.system,
    pytest.mark.withdata
]


def test_telemetry_main_withdata_run(system, system_config, system_files, sim_output):
    if not system_config.get("telemetry", False):
        pytest.skip(f"{system} does not support telemetry run.")
    if not system_config.get("withdata", False):
        pytest.skip(f"{system} does not support telemetry run with data.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "telemetry",
        "--system", system,
        "-f", ','.join(system_files),
        "-o", sim_output,
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
