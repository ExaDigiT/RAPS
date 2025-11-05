import pytest
from ..util import run_engine


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
]


@pytest.mark.parametrize("start", [
    "2025-01-01", "2024-01-04T00:00Z", "1970-01-01T00:00:00+00:00",
])
def test_main_start_run(system, system_config, sim_output, start):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main even without data.")

    engine, stats = run_engine({
        "system": system,
        "time": "10s",
        "start": start
    })
