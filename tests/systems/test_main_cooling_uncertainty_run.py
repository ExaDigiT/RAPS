import os
import subprocess
import pytest
from tests.util import PROJECT_ROOT

pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.cooling,
    pytest.mark.uncertainty
]


def test_main_cooling_uncertainty_run(request, system, system_config, sim_output):
    print(f"Markexpr: {request.config.option.markexpr}")
    if not system_config.get("uncertainty", False) or not system_config.get("cooling", False):
        pytest.skip(f"{system} does not support cooling or uncertainty.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "--time", "3m",
        "--system", system,
        "-c",
        "-u",
        "--noui",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
