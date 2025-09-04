import glob
import fnmatch
import functools
from typing import Any, Literal
from pathlib import Path
from functools import cached_property
import yaml
from pydantic import BaseModel, computed_field, model_validator, field_validator
from raps.raps_config import raps_config

# Define Pydantic models for the config to handle parsing and validation


class SystemSystemConfig(BaseModel):
    num_cdus: int
    racks_per_cdu: int
    nodes_per_rack: int
    chassis_per_rack: int
    nodes_per_blade: int
    switches_per_chassis: int
    nics_per_node: int
    rectifiers_per_chassis: int
    nodes_per_rectifier: int
    missing_racks: list[int] = []
    down_nodes: list[int] = []
    cpus_per_node: int
    gpus_per_node: int
    cpu_peak_flops: float
    gpu_peak_flops: float
    cpu_fp_ratio: float
    gpu_fp_ratio: float
    threads_per_core: int | None = None
    cores_per_cpu: int | None = None

    @model_validator(mode='after')
    def _update_down_nodes(self):
        for rack in self.missing_racks:
            start_node_id = rack * self.nodes_per_rack
            end_node_id = start_node_id + self.nodes_per_rack
            self.down_nodes.extend(range(start_node_id, end_node_id))
        self.down_nodes = sorted(set(self.down_nodes))
        return self

    @computed_field
    @cached_property
    def num_racks(self) -> int:
        return self.num_cdus * self.racks_per_cdu - len(self.missing_racks)

    @computed_field
    @cached_property
    def sc_shape(self) -> list[int]:
        return [self.num_cdus, self.racks_per_cdu, self.nodes_per_rack]

    @computed_field
    @cached_property
    def total_nodes(self) -> int:
        return self.num_cdus * self.racks_per_cdu * self.nodes_per_rack

    @computed_field
    @cached_property
    def blades_per_chassis(self) -> int:
        return int(self.nodes_per_rack / self.chassis_per_rack / self.nodes_per_blade)

    @computed_field
    @cached_property
    def power_df_header(self) -> list[str]:
        power_df_header = ["CDU"]
        for i in range(1, self.racks_per_cdu + 1):
            power_df_header.append(f"Rack {i}")
        power_df_header.append("Sum")
        for i in range(1, self.racks_per_cdu + 1):
            power_df_header.append(f"Loss {i}")
        power_df_header.append("Loss")
        return power_df_header

    @computed_field
    @cached_property
    def available_nodes(self) -> int:
        return self.total_nodes - len(self.down_nodes)


class SystemPowerConfig(BaseModel):
    power_gpu_idle: float
    power_gpu_max: float
    power_cpu_idle: float
    power_cpu_max: float
    power_mem: float
    power_nic: float | None = None
    power_nic_idle: float | None = None
    power_nic_max: float | None = None
    power_nvme: float
    power_switch: float
    power_cdu: float
    power_update_freq: int
    rectifier_peak_threshold: float
    sivoc_loss_constant: float
    sivoc_efficiency: float
    rectifier_loss_constant: float
    rectifier_efficiency: float
    power_cost: float


class SystemUqConfig(BaseModel):
    power_gpu_uncertainty: float
    power_cpu_uncertainty: float
    power_mem_uncertainty: float
    power_nic_uncertainty: float
    power_nvme_uncertainty: float
    power_cdus_uncertainty: float
    power_node_uncertainty: float
    power_switch_uncertainty: float
    rectifier_power_uncertainty: float


JobEndStates = Literal["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]


class SystemSchedulerConfig(BaseModel):
    job_arrival_time: int
    mtbf: int
    trace_quanta: int
    min_wall_time: int
    max_wall_time: int
    ui_update_freq: int  # TODO should be moved to raps_config
    max_nodes_per_job: int
    job_end_probs: dict[JobEndStates, float]
    multitenant: bool = False


class SystemCoolingConfig(BaseModel):
    cooling_efficiency: float
    wet_bulb_temp: float
    zip_code: str | None = None
    country_code: str | None = None
    fmu_path: str
    fmu_column_mapping: dict[str, str]
    w_htwps_key: str
    w_ctwps_key: str
    w_cts_key: str
    temperature_keys: list[str]


class SystemNetworkConfig(BaseModel):
    topology: Literal["capacity", "fat-tree", "dragonfly", "torus3d"]
    network_max_bw: float
    latency: float | None = None

    fattree_k: int | None = None

    dragonfly_d: int | None = None
    dragonfly_a: int | None = None
    dragonfly_p: int | None = None

    torus_x: int | None = None
    torus_y: int | None = None
    torus_z: int | None = None
    torus_wrap: bool | None = None
    torus_link_bw: float | None = None
    torus_routing: str | None = None

    hosts_per_router: int | None = None
    latency_per_hop: float | None = None
    node_coords_csv: str | None = None


class SystemConfig(BaseModel):
    system_name: str
    """ Name of the system, defaults to the yaml file name """

    system: SystemSystemConfig
    power: SystemPowerConfig
    scheduler: SystemSchedulerConfig
    uq: SystemUqConfig | None = None
    cooling: SystemCoolingConfig | None = None
    network: SystemNetworkConfig | None = None

    def get_legacy(self) -> dict[str, Any]:
        """
        Return the system config as a flattened, uppercased dict. This is for backwards
        compatibility with the rest of RAPS code so we can migrate to the new config format
        gradually. The dict also as a "system_config" key that contains the SystemConfig object
        itself.
        """
        renames = {  # fields that need to be renamed to something other than just .upper()
            "system_name": "system_name",
            "w_htwps_key": "W_HTWPs_KEY",
            "w_ctwps_key": "W_CTWPs_KEY",
            "w_cts_key": "W_CTs_KEY",
            "multitenant": "multitenant",
        }
        dump = self.model_dump(mode="json", exclude_none=True)

        config_dict: dict[str, Any] = {}
        for k, v in dump.items():  # flatten
            if isinstance(v, dict):
                config_dict.update(v)
            else:
                config_dict[k] = v
        # rename keys
        config_dict = {renames.get(k, k.upper()): v for k, v in config_dict.items()}
        config_dict['system_config'] = self
        return config_dict


class MultiPartitionSystemConfig(BaseModel):
    system_name: str
    partitions: list[SystemConfig]

    @field_validator("partitions")
    def _validate_partitions(cls, partitions: list[SystemConfig]):
        partition_names = [c.system_name for c in partitions]
        if len(set(partition_names)) != len(partition_names):
            raise ValueError(f"Duplicate system names: {','.join(partition_names)}")
        return partitions

    @property
    def partition_names(self):
        return [c.system_name for c in self.partitions]


@functools.cache
def list_systems() -> list[str]:
    """ Lists all available systems """
    return sorted([
        str(p.relative_to(raps_config.system_config_dir)).removesuffix(".yaml")
        for p in raps_config.system_config_dir.rglob("*.yaml")
    ])


@functools.cache
def get_system_config(system: str) -> SystemConfig:
    """
    Returns the system config as a Pydantic object.
    system can either be a path to a custom .yaml file, or the name of one of the pre-configured
    systems defined in RAPS_SYSTEM_CONFIG_DIR.
    """
    if system in list_systems():
        config_path = raps_config.system_config_dir / f"{system}.yaml"
        system_name = system
    else:
        config_path = Path(system).resolve()
        system_name = config_path.stem

    if not config_path.is_file():
        raise FileNotFoundError(f'"{system}" not found. Valid systems are: {list_systems()}')
    config = {
        "system_name": system_name,  # You can override system_name in the yaml as well
        **yaml.safe_load(config_path.read_text()),
    }
    return SystemConfig.model_validate(config)


def get_partition_configs(partitions: list[str]) -> MultiPartitionSystemConfig:
    """
    Resolves multiple partition config files. Can pass globs, or directories to include all yaml
    files under the directory.
    """
    systems = list_systems()
    multi_partition_systems = set(s.split("/")[0] for s in systems if "/" in s)
    combined_system_name = []

    parsed_configs: list[SystemConfig] = []
    for pat in partitions:
        if pat in multi_partition_systems:
            matched_systems = fnmatch.filter(systems, f"{pat}/*")
            combined_system_name.append(pat)
        elif fnmatch.filter(systems, pat):
            matched_systems = fnmatch.filter(systems, pat)
            combined_system_name.extend(s.split("/")[0] for s in matched_systems)
        elif Path(pat).is_dir():
            matched_systems = sorted(Path(pat).glob("*.yaml"))
            combined_system_name.append(Path(pat).name)
        else:
            matched_systems = sorted(glob.glob(pat))
            combined_system_name.extend(Path(s).stem for s in matched_systems)

        if not matched_systems:
            raise FileNotFoundError(f'No config files match "{pat}"')
        parsed_configs.extend(get_system_config(s) for s in sorted(matched_systems))

    if len(parsed_configs) == 1:
        combined_system_name = parsed_configs[0].system_name
    else:
        combined_system_name = "+".join(dict.fromkeys(combined_system_name))  # dedup, keep order
    return MultiPartitionSystemConfig(
        system_name=combined_system_name,
        partitions=parsed_configs,
    )
