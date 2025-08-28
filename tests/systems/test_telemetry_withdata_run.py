import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT, DATA_PATH


pytestmark = [
    pytest.mark.system,
    pytest.mark.withdata
]


def test_telemetry_main_withdata_run(system, system_config, system_file, sim_output):
    if not system_config.get("telemetry", False):
        pytest.skip(f"{system} does not support telemetry run.")
    if not system_config.get("withdata", False):
        pytest.skip(f"{system} does not support telemetry run with data.")

    if isinstance(system_file, list):
        file_list = [DATA_PATH / system / x for x in system_file]
    else:
        file_list = [DATA_PATH / system / system_file]
    for file in file_list:
        assert os.path.isfile(file) or os.path.isdir(file), \
            f"File `{file}' does not exist. does ./data exist or is RAPS_DATA_DIR set?"
    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "telemetry",
        "--system", system,
        "-f", *file_list,
        "-o", sim_output,
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
