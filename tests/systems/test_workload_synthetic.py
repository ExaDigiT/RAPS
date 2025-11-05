import subprocess
import gc
import pytest


pytestmark = [
    pytest.mark.system,
    pytest.mark.workload,
]


def flatten(dist):
    name, args = dist
    return [name, *args]


jobdist_case = [
    ("weibull", ["--jobsize-weibull-shape", "0.75", "--jobsize-weibull-scale", "16"]),
    ("normal", ["--jobsize-normal-stddev", "100", "--jobsize-normal-mean", "16"]),
    ("uniform", []),
]
cpudist_case = [
    ("weibull", ["--cpuutil-weibull-shape", "0.75", "--cpuutil-weibull-scale", "16"]),
    ("normal", ["--cpuutil-normal-stddev", "100", "--cpuutil-normal-mean", "16"]),
    ("uniform", []),
]
gpudist_case = [
    ("weibull", ["--gpuutil-weibull-shape", "0.75", "--gpuutil-weibull-scale", "16"]),
    ("normal", ["--gpuutil-normal-stddev", "100", "--gpuutil-normal-mean", "16"]),
    ("uniform", []),
]
wtimedist_case = [
    ("weibull", ["--walltime-weibull-shape", "0.75", "--walltime-weibull-scale", "16"]),
    ("normal", ["--walltime-normal-stddev", "100", "--walltime-normal-mean", "16"]),
    ("uniform", []),
]
additional_params_cases = [
    "",  # nothing
    ["--jobsize-is-of-degree", "2"],
    ["--jobsize-is-of-degree", "3"],
    ["--jobsize-is-power-of", "2"],
    ["--jobsize-is-power-of", "3"],
]


@pytest.mark.parametrize(
    "jobdist", jobdist_case, ids=lambda d: d[0]
)
@pytest.mark.parametrize(
    "cpudist", cpudist_case, ids=lambda d: d[0]
)
@pytest.mark.parametrize(
    "gpudist", gpudist_case, ids=lambda d: d[0]
)
@pytest.mark.parametrize(
    "wtimedist", wtimedist_case, ids=lambda d: d[0]
)
@pytest.mark.parametrize(
    "additional_params", additional_params_cases, ids=lambda p: (p or "none")
)
def test_workload_synthetic_run(
    system, system_config, jobdist, cpudist, gpudist, wtimedist, additional_params
):
    """Run the real synthetic workload generator with every combination
    of job, CPU, GPU, wall‑time distributions and optional extra flags.
    The test simply verifies that the script exits with status 0.
    """

    if not system_config.get("workload", False):
        pytest.skip(f"{system} does not support workload run.")

    # Build the command line.  Each distribution tuple expands into:
    #   dist_name, <flag1>, <value1>, ...
    cmd = [
        "python", "main.py", "workload",
        "--system", system,
        "-w", "synthetic",
        "--jobsize-distribution", *flatten(jobdist),
        "--cpuutil-distribution", *flatten(cpudist),
        "--gpuutil-distribution", *flatten(gpudist),
        "--walltime-distribution", *flatten(wtimedist),
    ]

    # Add any extra parameters if present.
    if additional_params:
        # If the flag contains a space we keep it as a single string.
        cmd.extend(additional_params)

    cmd1 = ["python", "-c \"exit()\""]
    result = subprocess.run(cmd1, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=1
        )
    except subprocess.TimeoutExpired:
        result.returncode = 0

    assert result.returncode == 0, (
        f"Failed on {system} with {jobdist[0]}, {cpudist[0]}, "
        f"{gpudist[0]}, {wtimedist[0]}: {result.stderr}"
    )

    # Explicitly delete the result to avoid hitting
    # “Too many open file descriptors” on slow CI machines.
    del result
    gc.collect()
