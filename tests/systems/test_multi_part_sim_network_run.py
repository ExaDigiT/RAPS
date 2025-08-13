import os
import subprocess
import gc
import pytest
from tests.util import PROJECT_ROOT


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata
]


def test_multi_part_sim_run(system, system_config, random_id):

    if not system_config.get("multi-part-sim", False):
        pytest.skip(f"{system} does not support basic multi-part-sim run.")

    if not system_config.get("net", False):
        pytest.skip(f"{system} does not support network run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "multi-part-sim.py",
        "--time", "1h",
        "--system", system,
        "-x", f"{system}/*",
        "-net",
        #"--noui"
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"

    #TODO:
    #Cleanup files after test!

    del result
    gc.collect()
