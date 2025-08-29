import os
import subprocess
import gc
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.uncertainty,
    pytest.mark.long
]


def test_main_uncertainty_run(system, system_config, random_id):
    if not system_config.get("uncertainty", False):
        pytest.skip(f"{system} does not support uncertainty.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", "3m",
        "--system", system,
        "-u",
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
