import pytest
from ..util import run_engine

pytestmark = [
    pytest.mark.system,
    pytest.mark.withdata,
    pytest.mark.long
]


def test_main_withdata_run(system, system_config, system_files, sim_output):
    if not system_config.get("main", False):
        pytest.skip(f"{system} does not support basic main even without data.")
    if not system_config.get("withdata", False):
        pytest.skip(f"{system} does not support basic main with data.")

    engine, stats = run_engine({
        "system": system,
        "time": "20m",
        "replay": system_files,
    })

    # Check that it at least loaded some data
    assert stats['tick_count'] == 20 * 60
    assert stats['job']['jobs_total'] > 0
    assert len(stats['job']['jobs_still_running']) + stats['job']['jobs_completed'] > 0
