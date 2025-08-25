from pathlib import Path
from raps.utils import ExpandedPath
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
ROOT_DIR = Path(__file__).parent.parent


class RapsConfig(BaseSettings):
    """
    General settings for raps. Pydantic will automatically populate this model from env vars or a
    .env file.
    """
    # TODO I think we should move more of general/ui related settings from SimConfig into here.
    # We'll be using SimConfig in the simulation server and those settings aren't applicable there,
    # so it makes sense to keep SimConfig scoped to the logical operation of the sim.

    system_config_dir: ExpandedPath = ROOT_DIR / 'config'
    """ Directory containing system configuration files """

    model_config = SettingsConfigDict(
        yaml_file="raps_config.yaml",
        env_prefix='raps_',
        env_nested_delimiter='__',
        nested_model_default_partial_update=True,
    )

    # Customize setting sources, we'll use yaml config file instead of the default .env
    @classmethod
    def settings_customise_sources(
        cls, settings_cls,
        init_settings, env_settings, dotenv_settings, file_secret_settings,
    ):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls),)


raps_config = RapsConfig()
