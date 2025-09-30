import pytest
from ..util import run_engine


pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata,
    pytest.mark.fastforward
]


@pytest.mark.parametrize("ff_arg", ["0s", "1s", "3600s", "60m"])
def test_main_fastforward_run(system, system_config, ff_arg, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main even without data.")

    engine, stats = run_engine({
        "system": system,
        "fastforward": ff_arg,
        "time": "10s",
    })
    assert stats['engine']['time_simulated'] == '0:00:10'
