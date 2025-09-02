import os
import subprocess
import gc
import pytest
from tests.util import PROJECT_ROOT
from raps.utils import convert_seconds_to_hhmmss, parse_td


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.time_delta
]


@pytest.mark.parametrize("time_arg, tdelta_arg", [
    ("10", "1ds"),
    ("60", "3ds"),
    ("1", "1cs"),
    ("1", "1ms"),
    ("10ds", "1cs"),
    ("10cs", "1ms"),
    ("100ms", "1ms"),
    ("100ms", "1s"),
], ids=["1ds", "3ds", "1cs", "1ms", "1cs-for-10ds", "1ms-for-10cs", "1ms-for-100ms", "1s-for-100ms"])
def test_main_time_delta_sub_second_run(system, system_config, time_arg, tdelta_arg, sim_output):
    if not system_config.get("time_delta", False):
        pytest.skip(f"{system} does not support time_delta run.")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run([
        "python", "main.py", "run",
        "-t", time_arg,
        "--time-delta", tdelta_arg,
        "--system", system,
        "--noui",
        "-o", sim_output
    ], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, f"Failed on {system}: {result.stderr}"
    time = parse_td(time_arg).seconds
    assert f"Time Simulated: {convert_seconds_to_hhmmss(time)}" in result.stdout

    subprocess.run(
        f"rm {sim_output}.npz && rm -fr simulation_results/{sim_output}",
        shell=True,
        check=True
    )

    del result
    gc.collect()
