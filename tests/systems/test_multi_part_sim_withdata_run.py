import pytest
from tests.util import run_multi_part_engine


pytestmark = [
    pytest.mark.system,
    pytest.mark.withdata,
    pytest.mark.long
]


def test_multi_part_sim_withdata_run(system, system_config, system_files, sim_output):
    if not system_config.get("multi-part-sim", False):
        pytest.skip(f"{system} does not support basic multi-part-sim run even without data.")
    if not system_config.get("withdata", False):
        pytest.skip(f"{system} does not support multi-part-sim run with data.")

    engine, stats = run_multi_part_engine({
        "start": system_config['start'],
        "time": "1h",
        "partitions": [system],
        "replay": system_files,
    })

    times = [s['engine']['time_simulated'] for s in stats['partitions'].values()]
    assert len(set(times)) == 1  # All run the same time
