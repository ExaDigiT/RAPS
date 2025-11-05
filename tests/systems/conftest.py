import pytest
from tests.util import DATA_PATH


SYSTEM_CONFIGS = {
    "40frontiers": {
        "marks": [pytest.mark.long],  # All these tests are long running as the system is large.
        "main": True,
        "telemetry": False,
        "workload": False,
        "multi-part-sim": False,
        "withdata": False,
        "start": None,
        "files": [],
        "cooling": False,
        "uncertainty": True,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "adastraMI250": {
        "marks": [],
        "main": True,
        "telemetry": True,
        "workload": True,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2024-09-01T02:00:00Z",
        "files": ["adastraMI250/AdastaJobsMI250_15days.parquet"],
        "cooling": False,
        "uncertainty": True,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "bluewaters": {
        "marks": [],
        "main": True,
        "telemetry": True,
        "workload": True,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2017-03-28T02:00:00Z",
        "files": ["bluewaters"],
        "cooling": False,
        "uncertainty": False,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "frontier": {
        "marks": [],
        "main": True,
        "telemetry": True,
        "workload": True,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2024-01-18T03:00:00Z",
        "files": ["frontier/slurm/joblive/date=2024-01-18/", "frontier/jobprofile/date=2024-01-18/"],
        "cooling": True,
        "uncertainty": True,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "fugaku": {
        "marks": [],
        "main": True,
        "telemetry": True,
        "workload": True,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2021-04-03T02:00:00Z",
        "files": ["fugaku/21_04.parquet"],
        "cooling": False,
        "uncertainty": False,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "gcloudv2": {
        "marks": [],
        "main": True,
        "telemetry": True,
        "workload": True,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2011-05-02T05:00:00Z",
        "files": ["gcloud/v2/google_cluster_data_2011_sample"],
        "cooling": False,
        "uncertainty": False,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "lassen": {
        "marks": [],
        "main": True,
        "telemetry": False,  # Takes very long!
        "workload": False,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2019-08-22T00:00:00Z",
        "files": ["lassen/Lassen-Supercomputer-Job-Dataset"],
        "cooling": True,
        "uncertainty": False,
        "time": True,
        "time_delta": True,
        "net": True,
    },
    "marconi100": {
        "marks": [],
        "main": True,
        "telemetry": True,
        "workload": True,
        "multi-part-sim": False,
        "withdata": True,
        "start": "2020-05-06T07:30:00Z",
        "files": ["marconi100/job_table.parquet"],
        "cooling": True,
        "uncertainty": False,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "mit_supercloud": {
        "marks": [],
        "main": False,
        "telemetry": False,
        "workload": False,
        "multi-part-sim": True,
        "withdata": True,
        "start": "2021-05-22T00:00:00Z",
        "files": ["mit_supercloud/202201"],
        "cooling": False,
        "uncertainty": False,
        "time": False,
        "time_delta": False,
        "net": False,
        "net-multi-sim": True,
    },
    "setonix": {
        "marks": [],
        "main": False,
        "telemetry": True,
        "workload": False,
        "multi-part-sim": True,
        "withdata": False,
        "files": [],
        "start": None,
        "cooling": False,
        "uncertainty": False,
        "time": False,
        "time_delta": False,
        "net": False,
    },
    "summit": {
        "marks": [],
        "main": True,
        "telemetry": False,
        "workload": False,
        "multi-part-sim": False,
        "withdata": False,
        "files": [],
        "start": None,
        "cooling": True,
        "uncertainty": False,
        "time": True,
        "time_delta": True,
        "net": False,
    },
    "lumi": {
        "marks": [],
        "main": False,
        "telemetry": False,
        "workload": False,
        "multi-part-sim": True,
        "withdata": False,
        "files": [],
        "start": None,
        "cooling": False,
        "uncertainty": False,
        "time": False,
        "time_delta": False,
        "net": False,
        "net-multi-sim": False
    },
}


@pytest.fixture(params=[
    pytest.param(k, marks=v.get('marks', [])) for k, v in SYSTEM_CONFIGS.items()
])
def system(request):
    return request.param


# Add markers to each test for the system.
# Similar to pytest -m marker.
# These are explicitly defined in pytest.ini, to avoid warnings.
# This way you can run test with: pytest -m systemname
def pytest_collection_modifyitems(config, items):
    for item in items:
        system = item.callspec.params.get("system") if hasattr(item, "callspec") else None
        if system:
            item.add_marker(getattr(pytest.mark, system))


# #Define tests to run here!
@pytest.fixture
def system_config(system):
    return SYSTEM_CONFIGS[system]


@pytest.fixture
def system_files(system):
    file_list = [DATA_PATH / f for f in SYSTEM_CONFIGS[system].get('files', [])]
    for file in file_list:
        assert file.exists(), \
            f"File `{file}' does not exist. does ./data exist or is RAPS_DATA_DIR set?"

    return [str(f) for f in file_list]
