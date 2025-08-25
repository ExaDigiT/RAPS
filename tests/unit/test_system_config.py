import pytest
from raps.raps_config import raps_config
from raps.system_config import list_systems, get_system_config, get_partition_configs


@pytest.mark.parametrize("system_name", list_systems())
def test_configs(system_name):
    # Very basic test that all system configs are valid
    config = get_system_config(system_name)
    assert config.system_name == system_name
    assert config.get_legacy()['system_name'] == system_name
    assert config.get_legacy()['system_config'] == config


@pytest.mark.parametrize("input,expected_name,expected_configs", [
    (["lumi"], "lumi", ["lumi/lumi-c", "lumi/lumi-g"]),
    (["lumi/*"], "lumi", ["lumi/lumi-c", "lumi/lumi-g"]),
    (["frontier", "summit"], "frontier+summit", ["frontier", "summit"]),
    # test passing arbitrary paths
    ([str(raps_config.system_config_dir / "lumi")], "lumi", ["lumi-c", "lumi-g"]),
    ([str(raps_config.system_config_dir / "lumi/lumi-*")], "lumi-c+lumi-g", ["lumi-c", "lumi-g"]),
])
def test_get_partition_configs(input, expected_name, expected_configs):
    result = get_partition_configs(input)
    assert result.system_name == expected_name
    assert result.partition_names == expected_configs
