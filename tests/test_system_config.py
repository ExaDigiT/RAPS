import pytest
from raps.config import list_systems, get_system_config

@pytest.mark.parametrize("system_name", list_systems())
def test_configs(system_name):
    # Very basic test that all system configs are valid
    config = get_system_config(system_name)
    assert config.system_name == system_name
    assert config.get_legacy()['system_name'] == system_name
    assert config.get_legacy()['config'] == config
