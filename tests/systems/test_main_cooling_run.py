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


def test_main_cooling_run(system, system_config, random_id):
    if not system_config.get("cooling", False):
        pytest.skip(f"{system} does not support cooling.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", "1h",
        "--system", system,
        "-c",
        "--noui",
        "-o", random_id
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"

    subprocess.run(
        f"rm {random_id}.npz && rm -fr simulation_results/{random_id}",
        shell=True,
        check=True
    )

    del result
    gc.collect()
