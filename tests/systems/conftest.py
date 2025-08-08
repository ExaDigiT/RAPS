import pytest


@pytest.fixture(params=[
    pytest.param("40frontiers", marks=pytest.mark.long),  # All these tests are long running as the system is large.
    "adastraMI250",
    "frontier",
    "fugaku",
    "gcloudv2",
    "lassen",
    "marconi100",
    "mit_supercloud",
    "setonix",
    "summit",
    "lumi"
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
    # Defaults for systems not listed explicitly
    default_config = {}  # No defaults!

    configs = {
        "40frontiers": {
            "basic": True,
            "multi-part-sim": False,
            "withdata": False,
            "cooling": False,
            "uncertainty": True,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "adastraMI250": {
            "basic": True,
            "multi-part-sim": False,
            "withdata": True,
            "cooling": False,
            "uncertainty": True,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "frontier": {
            "basic": True,
            "multi-part-sim": False,
            "withdata": True,
            "cooling": True,
            "uncertainty": True,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "fugaku": {
            "basic": True,
            "multi-part-sim": False,
            "withdata": True,
            "cooling": False,
            "uncertainty": False,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "gcloudv2": {
            "basic": False,
            "multi-part-sim": False,
            "withdata": False,
            "cooling": False,
            "uncertainty": False,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "lassen":{
            "basic": True,
            "multi-part-sim": False,
            "withdata": True,
            "cooling": True,
            "uncertainty": False,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": True,
        },
        "marconi100":{
            "basic": True,
            "multi-part-sim": False,
            "withdata": True,
            "cooling": True,
            "uncertainty": False,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "mit_supercloud": {
            "basic": False,
            "multi-part-sim": True,
            "withdata": True,
            "cooling": False,
            "uncertainty": False,
            "time": False,
            "fastforward": False,
            "time_delta": False,
            "net": False,
            "net-multi-sim": True,
        },
        "setonix": {
            "basic": False,
            "multi-part-sim": True,
            "withdata": False,
            "cooling": False,
            "uncertainty": False,
            "time": False,
            "fastforward": False,
            "time_delta": False,
            "net": False,
        },
        "summit": {
            "basic": True,
            "multi-part-sim": False,
            "withdata": False,
            "cooling": True,
            "uncertainty": False,
            "time": True,
            "fastforward": True,
            "time_delta": True,
            "net": False,
        },
        "lumi": {
            "basic": False,
            "multi-part-sim": True,
            "withdata": False,
            "cooling": False,
            "uncertainty": False,
            "time": False,
            "fastforward": False,
            "time_delta": False,
            "net": False,
            "net-multi-sim": False
        },


    }
    return configs.get(system, default_config)


@pytest.fixture
def system_file(system):
    files = {
        "40frontiers":[],
        "adastraMI250":["AdastaJobsMI250_15days.parquet"],
        "frontier":["slurm/joblive/date=2024-01-18/","jobprofile/date=2024-01-18/"],
        "fugaku":["21_04.parquet"],
        "gcloudv2":["/v2/google_cluster_data_2011_sample"],
        "lassen":["Lassen-Supercomputer-Job-Dataset"],
        "marconi100":["job_table.parquet"],
        "mit_supercloud":["202201"],
        "setonix":[""],
        "summit":[],
        "lumi":[]
    }
    return files.get(system,files)
