"""Tests for the parsed-replay snapshot cache (raps/cache.py, Telemetry.load_data_cached)."""
from datetime import datetime, timezone
import pytest
from raps.job import Job, job_dict
from raps.telemetry import Telemetry
from raps.utils import WorkloadData

LOADS = []  # one entry per call to load_data, so tests can tell parses from cache hits


def load_data(files, **kwargs):
    """ Stand-in dataloader; this module is passed as `dataloader` below. """
    LOADS.append(kwargs.get("jid"))
    jobs = [Job(job_dict(nodes_required=2, name=f"job{i}", account="acct", id=i, submit_time=i,
                         start_time=i, end_time=i + 10, time_limit=10, expected_run_time=10,
                         cpu_trace=[1.0], gpu_trace=[1.0], ntx_trace=[], nrx_trace=[]))
            for i in range(3)]
    return WorkloadData(jobs=jobs, telemetry_start=0, telemetry_end=100,
                        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc))


@pytest.fixture()
def data_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPS_CACHE_DIR", str(tmp_path / "cache"))
    LOADS.clear()
    f = tmp_path / "telemetry.parquet"
    f.write_text("raw data")
    return f


def telemetry(**kwargs):
    return Telemetry(system="fake", config={}, arrival="prescribed",
                     dataloader=__name__, **kwargs)


def test_second_load_hits_cache(data_file):
    first = telemetry().load_from_files([data_file])
    second = telemetry().load_from_files([data_file])
    assert len(LOADS) == 1
    assert [j.id for j in second.jobs] == [j.id for j in first.jobs]
    assert second.telemetry_end == first.telemetry_end and second.start_date == first.start_date


def test_loader_options_and_inputs_invalidate(data_file):
    telemetry(jid="1").load_from_files([data_file])
    telemetry(jid="2").load_from_files([data_file])  # different loader option
    assert len(LOADS) == 2
    data_file.write_text("changed raw data")  # different input
    telemetry(jid="1").load_from_files([data_file])
    assert len(LOADS) == 3


def test_simulation_only_options_do_not_invalidate(data_file):
    telemetry(policy="fcfs").load_from_files([data_file])
    telemetry(policy="sjf", cooling=True).load_from_files([data_file])
    assert len(LOADS) == 1


def test_cache_modes(data_file):
    telemetry(cache="off").load_from_files([data_file])
    telemetry(cache="off").load_from_files([data_file])
    assert len(LOADS) == 2
    telemetry(cache="refresh").load_from_files([data_file])  # parses and writes
    telemetry().load_from_files([data_file])  # hit
    assert len(LOADS) == 3
    telemetry(cache="refresh").load_from_files([data_file])
    assert len(LOADS) == 4


def test_corrupt_entry_is_reparsed(data_file, tmp_path):
    telemetry().load_from_files([data_file])
    (entry,) = (tmp_path / "cache" / "snapshots").glob("*.npz")
    entry.write_bytes(b"garbage")
    telemetry().load_from_files([data_file])
    assert len(LOADS) == 2
    telemetry().load_from_files([data_file])  # entry was rewritten
    assert len(LOADS) == 2
