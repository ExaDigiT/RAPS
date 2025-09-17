import pytest
from ..util import run_engine

pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata
]


def test_engine_basic(system, system_config, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main run.")

    engine, stats = run_engine({
        "system": system,
        "time": "2m",
    })

    assert stats['tick_count'] == 120
    assert stats['engine']['time_simulated'] == '0:02:00'
