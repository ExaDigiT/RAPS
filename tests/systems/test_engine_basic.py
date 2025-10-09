import pytest
from ..util import run_engine
from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.stats import get_engine_stats, get_job_stats, RunningStats

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


def test_engine_stats(system, system_config, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main run.")

    engine = Engine(SingleSimConfig.model_validate({
        "system": system,
        "time": "2m",
    }))
    gen = engine.run_simulation()
    running_stats = RunningStats(engine)

    for tick in gen:
        stats = running_stats.get_stats()
    stats = running_stats.get_stats()

    final_stats = {
        **get_engine_stats(engine),
        **get_job_stats(engine),
    }

    # Confirm the running stats match up with the final stat computation
    for stat in stats.keys():
        assert pytest.approx(stats[stat]) == final_stats[stat], f"stat {stat}"
