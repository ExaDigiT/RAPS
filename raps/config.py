import os, functools
from typing import Any, Literal
from pathlib import Path
import yaml
from pydantic import BaseModel, computed_field, model_validator

ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = Path(os.environ.get("RAPS_CONFIG", ROOT_DIR / 'config')).resolve()

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
    threads_per_core: int|None = None
    cores_per_cpu: int|None = None

    @model_validator(mode='after')
    def _update_down_nodes(self):
        for rack in self.missing_racks:
            start_node_id = rack * self.nodes_per_rack
            end_node_id = start_node_id + self.nodes_per_rack
            self.down_nodes.extend(range(start_node_id, end_node_id))
        self.down_nodes = sorted(set(self.down_nodes))
        return self

    @computed_field
    @property
    def num_racks(self) -> int:
        return self.num_cdus * self.racks_per_cdu - len(self.missing_racks)

    @computed_field
    @property
    def sc_shape(self) -> list[int]:
        return [self.num_cdus, self.racks_per_cdu, self.nodes_per_rack]

    @computed_field
    @property
    def total_nodes(self) -> int:
        return self.num_cdus * self.racks_per_cdu * self.nodes_per_rack

    @computed_field
    @property
    def blades_per_chassis(self) -> int:
        return int(self.nodes_per_rack / self.chassis_per_rack / self.nodes_per_blade)

    @computed_field
    @property
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
    @property
    def available_nodes(self) -> int:
        return self.total_nodes - len(self.down_nodes)

class SystemPowerConfig(BaseModel):
    power_gpu_idle: float
    power_gpu_max: float
    power_cpu_idle: float
    power_cpu_max: float
    power_mem: float
    power_nic: float|None = None
    power_nic_idle: float|None = None
    power_nic_max: float|None = None
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
    ui_update_freq: int
    max_nodes_per_job: int
    job_end_probs: dict[JobEndStates, float]
    multitenant: bool = False

class SystemCoolingConfig(BaseModel):
    cooling_efficiency: float
    wet_bulb_temp: float
    zip_code: str|None = None
    country_code: str|None = None
    fmu_path: str
    fmu_column_mapping: dict[str, str]
    w_htwps_key: str
    w_ctwps_key: str
    w_cts_key: str
    temperature_keys: list[str]

class SystemNetworkConfig(BaseModel):
    topology: Literal["fat-tree", "dragonfly", "torus3d"]
    network_max_bw: float
    latency: float|None = None

    fattree_k: int|None = None

    dragonfly_d: int|None = None
    dragonfly_a: int|None = None
    dragonfly_p: int|None = None

    torus_x: int|None = None
    torus_y: int|None = None
    torus_z: int|None = None
    torus_wrap: bool|None = None
    torus_link_bw: float|None = None
    torus_routing: str|None = None

    hosts_per_router: int|None = None
    latency_per_hop: float|None = None
    node_coords_csv: str|None = None

class SystemConfig(BaseModel):
    system_name: str
    system: SystemSystemConfig
    power: SystemPowerConfig
    scheduler: SystemSchedulerConfig
    uq: SystemUqConfig|None = None
    cooling: SystemCoolingConfig|None = None
    network: SystemNetworkConfig|None = None

    def get_legacy(self) -> dict[str, Any]:
        """
        Return the system config as a flattened, uppercased dict. This is for backwards
        compatibility with the rest of RAPS code so we can migrate to the new config format
        gradually. The dict also as a "config" key that contains the SystemConfig object itself.
        """
        renames = { # fields that need to be renamed to something other than just .upper()
            "system_name": "system_name",
            "w_htwps_key": "W_HTWPs_KEY",
            "w_ctwps_key": "W_CTWPs_KEY",
            "w_cts_key": "W_CTs_KEY",
            "multitenant": "multitenant",
        }
        dump = self.model_dump(mode = "json", exclude_none = True)

        config_dict: dict[str, Any] = {}
        for k, v in dump.items(): # flatten
            if isinstance(v, dict):
                config_dict.update(v)
            else:
                config_dict[k] = v
        # rename keys
        config_dict = {renames.get(k, k.upper()): v for k, v in config_dict.items()}
        config_dict['config'] = self
        return config_dict


@functools.cache
def list_systems() -> list[str]:
    """ Lists all available systems """
    return sorted([
        str(p.relative_to(CONFIG_PATH)).removesuffix(".yaml")
        for p in CONFIG_PATH.rglob("*.yaml")
    ])


@functools.cache
def get_system_config(system: str) -> SystemConfig:
    """
    Returns the system config as a Pydantic object.
    system can either be a path to a custom .yaml file, or the name of one of the pre-configured
    systems defined in RAPS_CONFIG.
    """
    config_path = Path(system.removesuffix(".yaml") + ".yaml")
    if config_path.exists() or config_path.is_absolute():
        system_name = config_path.resolve()
    else: # assume it's a pre-configured system
        system_name = system.removesuffix(".yaml")
        config_path = CONFIG_PATH / config_path
    if not config_path.is_file():
        raise FileNotFoundError(
            f'"{system}" not found. Known systems are: {list_systems()}'
        )
    config = {
        "system_name": system_name,
        **yaml.safe_load(config_path.read_text()),
    }
    return SystemConfig.model_validate(config)
