import pytest
import uuid
import shutil
from glob import glob
from pathlib import Path
import gc


def pytest_addoption(parser):
    parser.addoption(
        "--runlong", action="store_true", default=False, help="Run long-running tests"
    )


def pytest_runtest_setup(item):
    if "long" in item.keywords and not item.config.getoption("--runlong"):
        # reason = f"Skipping {item.nodeid} because it requires --runlong"
        reason = "Skipping test because it requires --runlong"
        pytest.skip(reason)


@pytest.fixture()
def sim_output():
    """
    Handles cleaning up output from the sim.
    Can also be used even if you aren't outputing anything to run garbage collection after the sim.
    """
    out = f"test-output/test-{str(uuid.uuid4())[:8]}"
    yield out
    for file in glob(f"{out}*"):
        if Path(file).is_dir():
            shutil.rmtree(file)
        else:
            Path(file).unlink()

    # Also force a garbage collection to clean up memory after running a simulation
    gc.collect()
