import pytest
from raps.engine import Engine
from raps.sim_config import SimConfig
from raps.stats import (
    get_engine_stats,
    # get_job_stats,
    # get_scheduler_stats,
    # get_network_stats,
)

pytestmark = [
    pytest.mark.system,
    pytest.mark.nodata
]


def test_engine(system, system_config, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main run.")

    sim_config = SimConfig.model_validate({
        "system": system,
        "time": "2m",
    })
    engine, workload_data, time_delta = Engine.from_sim_config(sim_config)
    jobs = workload_data.jobs
    timestep_start = workload_data.telemetry_start
    timestep_end = workload_data.telemetry_end
    ticks = list(engine.run_simulation(jobs, timestep_start, timestep_end, time_delta))

    assert len(ticks) == 120

    engine_stats = get_engine_stats(engine)
    # job_stats = get_job_stats(engine)
    # scheduler_stats = get_scheduler_stats(engine)
    # network_stats = get_network_stats(engine)

    assert engine_stats['time simulated'] == '0:02:00'
    # TODO: More specific tests of values
