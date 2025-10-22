import argparse
import abc
from pathlib import Path
import pandas as pd
from functools import cached_property
from datetime import timedelta
from typing import Literal, Annotated as A
from annotated_types import Len
import importlib
from raps.schedulers.default import PolicyType, BackfillType
from raps.utils import (
    parse_time_unit, convert_to_time_unit, infer_time_unit, ResolvedPath, create_casename,
    RAPSBaseModel, AutoAwareDatetime, SmartTimedelta, yaml_dump,
)
from raps.system_config import (
    SystemConfig, get_partition_configs, get_system_config, list_systems, resolve_system_reference,
)
from pydantic import model_validator, Field, BeforeValidator

Distribution = Literal['uniform', 'weibull', 'normal']


class SimConfig(RAPSBaseModel, abc.ABC):
    cooling: bool = False
    """ Include the FMU cooling model """
    simulate_network: bool = False
    """ Include network model """
    weather: bool = False
    """
    Include weather information in the cooling model.
    Defaults to True if replay, False otherwise.
    """

    # Simulation runtime options
    start: AutoAwareDatetime | None = None
    """ Start of simulation """
    # Exclude end from serialization as it is redundant with time
    end: A[AutoAwareDatetime | None, Field(exclude=True)] = None
    """ End of simulation. Pass either `time` or `end`, not both. """
    time: SmartTimedelta = timedelta(hours=1)
    """
    Length of time to simulate (default seconds).
    Can pass a string like 123, 27m, 3h, 7d
    Pass either `time` or `end`, not both.
    """
    fastforward: SmartTimedelta = timedelta(seconds=0)
    """
    "Fast-forward" the simulation by time amount before starting. This is just a convenience
    shortcut for setting --start without having to recall the exact start date of the dataset.
    Can pass a string like 15s, 1m, 1h
    """
    time_delta: SmartTimedelta = timedelta(seconds=1)
    """
    Step size for the power simulation (default seconds).
    Can pass a string like 15s, 1m, 1h, 1ms
    """
    time_unit: A[timedelta, BeforeValidator(parse_time_unit)] = timedelta(seconds=1)
    """
    The base unit of the simulation, determining how often it will tick the job scheduler.
    """

    @cached_property
    def time_int(self) -> int:
        """ Return time as an int of time_unit """
        return int(self.time / self.time_unit)

    @cached_property
    def time_delta_int(self) -> int:
        """ Return time_delta as an int of time_unit """
        return int(self.time_delta / self.time_unit)

    @cached_property
    def downscale(self) -> int:
        return int(timedelta(seconds=1) / self.time_unit)

    numjobs: int = 100
    """ Number of jobs to schedule """

    uncertainties: bool = False
    """ Use float-with-uncertainties (much slower) """

    seed: int | None = None
    """ Set RNG seed for deterministic simulation """

    output: ResolvedPath | Literal['none'] | None = None
    """
    Where to output power, cooling, and loss models for later analysis.
    If omitted it will output to raps-output-<id> by default.
    Set to "none" to disable file output entirely.
    """

    _random_output: Path | None = None

    def get_output(self) -> Path | None:
        if self.output is None:  # by default, output to a random directory
            if not self._random_output:
                self._random_output = Path(create_casename("raps-output-")).resolve()
            return self._random_output
        elif self.output == "none":  # allow explicitly disabling output with "none"
            return None
        else:
            return self.output  # return user defined output path

    debug: bool = False
    """ Enable debug mode and disable rich layout """
    noui: bool = False
    """ Run without UI """
    verbose: bool = False
    """ Enable verbose output """
    layout: Literal["layout1", "layout2"] = "layout1"
    """ UI layout """
    plot: list[Literal["power", "loss", "pue", "temp", "util"]] | None = None
    """ Plots to generate """

    imtype: Literal["png", "svg", "jpg", "pdf", "eps"] = "png"
    """ Plot image type """

    replay: list[ResolvedPath] | None = None
    """ Either: path/to/joblive path/to/jobprofile OR filename.npz """

    encrypt: bool = False
    """ Encrypt sensitive data in telemetry """

    power_scope: Literal['node', 'chip'] = "chip"
    """ node mode will use node power instead of CPU/GPU utilizations """

    jid: str = "*"
    """ Replay job id """

    scale: int = 0
    """ Scale telemetry to a smaller target system, --scale 192 """

    live: bool = False
    """ Grab data from live system. """

    # Workload arguments (TODO split into separate model)
    workload: Literal['random', 'benchmark', 'peak', 'idle', 'synthetic', 
                      'multitenant', 'replay', 'randomAI', 'calculon'] = "random"

    """ Type of synthetic workload """
    multimodal: list[float] = [1.0]
    """
    Percentage to draw from each distribution (list of floats). e.g. '0.2 0.8' percentages apply
    in order to the list of the  --distribution argument list.
    """
    # Jobsize
    jobsize_distribution: list[Distribution] | None = None
    """ Distribution type """
    jobsize_normal_mean: float | None = None
    """ Mean (mu) for Normal distribution """
    jobsize_normal_stddev: float | None = None
    """ Standard deviation (sigma) for Normal distribution """
    jobsize_weibull_shape: float | None = None
    """ Jobsize shape of weibull """
    jobsize_weibull_scale: float | None = None
    """ Jobsize scale of weibull """
    jobsize_is_of_degree: int | None = None
    """ Draw jobsizes from distribution of degree N (squared,cubed). """
    jobsize_is_power_of: int | None = None
    """ Draw jobsizes from distribution of power of N (2->2^x,3->3^x). """

    # Walltime
    walltime_distribution: list[Distribution] | None = None
    """ Distribution type """
    walltime_normal_mean: float | None = None
    """ Walltime mean (mu) for Normal distribution """
    walltime_normal_stddev: float | None = None
    """ Walltime standard deviation (sigma) for Normal distribution """
    walltime_weibull_shape: float | None = None
    """ Walltime shape of weibull """
    walltime_weibull_scale: float | None = None
    """ Walltime scale of weibull """
    # Utilizations (TODO should probably make a reusable "Distribution" submodel)
    cpuutil_distribution: list[Distribution] = ['uniform']
    """ Distribution type """
    cpuutil_normal_mean: float | None = None
    """ Walltime mean (mu) for Normal distribution """
    cpuutil_normal_stddev: float | None = None
    """ Walltime standard deviation (sigma) for Normal distribution """
    cpuutil_weibull_shape: float | None = None
    """ Walltime shape of weibull """
    cpuutil_weibull_scale: float | None = None
    """ Walltime scale of weibull """
    gpuutil_distribution: list[Distribution] = ['uniform']
    """ Distribution type """
    gpuutil_normal_mean: float | None = None
    """ Walltime mean (mu) for Normal distribution """
    gpuutil_normal_stddev: float | None = None
    """ Walltime standard deviation (sigma) for Normal distribution """
    gpuutil_weibull_shape: float | None = None
    """ Walltime shape of weibull """
    gpuutil_weibull_scale: float | None = None
    """ Walltime scale of weibull """
    gantt_nodes: bool = False
    """ Print Gannt with nodes required as line thickness (default false) """

    # Synthetic workloads
    scheduler: Literal[
        "default",
        "experimental",
        "fastsim",
        "multitenant",
        "scheduleflow",
    ] = "default"
    """ Scheduler name """
    policy: str | None = None
    """ Schedule policy """
    backfill: str | None = None
    """ Backfill policy """

    # Arrival
    arrival: Literal["prescribed", "poisson"] = "prescribed"
    """ Modify arrival distribution (poisson) or use original submit times (prescribed) """
    job_arrival_time: int | None = None
    """ Poisson arrival (seconds). Overrides system config scheduler.job_arrival_time """
    job_arrival_rate: float | None = None  # TODO define default here
    """ Modify Poisson rate (default 1) """

    # Accounts
    accounts: bool = False
    accounts_json: ResolvedPath | None = None
    """ Path to accounts JSON file from previous run """

    # Downtime
    downtime_first: SmartTimedelta | None = None
    """
    First downtime. Can pass a string like 27m, 3h, 7d
    """
    downtime_interval: SmartTimedelta | None = None
    """
    Interval between downtimes. Can pass a string like 123, 27m, 3h, 7d
    """
    downtime_length: SmartTimedelta | None = None
    """
    Downtime length. Can pass a string like 123, 27m, 3h, 7d
    """

    @cached_property
    def downtime_first_int(self) -> int | None:
        return None if self.downtime_first is None else int(self.downtime_first / self.time_unit)

    @cached_property
    def downtime_interval_int(self) -> int | None:
        return None if self.downtime_interval is None else int(self.downtime_interval / self.time_unit)

    @cached_property
    def downtime_length_int(self) -> int | None:
        return None if self.downtime_length is None else int(self.downtime_length / self.time_unit)

    # Continous Job Generation
    continuous_job_generation: bool = False
    """ Activate continuous job generation """
    maxqueue: int = 50
    """ Specify the max queue length for continuous job generation """

    filter: str | None = None
    """job filter \"traffic > 1e8\" """

    @model_validator(mode="before")
    def _validate_before(cls, data):
        # This is called with the raw input, before Pydantic parses it, so data is just a dict and
        # contain any data types.
        data = {**data}

        # infer time_unit
        td_fields = [
            "time_delta", "time", "fastforward",
            "downtime_first", "downtime_interval", "downtime_length",
        ]
        # infer time unit from other timedelta fields if it wasn't set explicitly
        if data.get('time_unit') is None:
            time_unit = min(
                [infer_time_unit(data[f]) for f in td_fields if data.get(f)],
                default=timedelta(seconds=1)
            )
        else:
            time_unit = parse_time_unit(data['time_unit'])
        data['time_unit'] = time_unit

        return data

    @model_validator(mode="after")
    def _validate_after(self):
        # Allow setting either start/end or start/time for backwards compatibility and convenience
        if self.start and self.fastforward:
            raise ValueError("start and fastforward are mutually exclusive")

        if self.start:
            self.start = pd.Timestamp(self.start).floor(self.time_unit).to_pydatetime()
        if self.end:
            self.end = pd.Timestamp(self.end).floor(self.time_unit).to_pydatetime()

        if self.end:
            if not self.start:
                raise ValueError("end requires start to be set")
            if 'time' not in self.model_fields_set:  # If time was not explicitly set
                self.time = self.end - self.start
        elif self.start:
            self.end = self.start + self.time

        if self.start and self.start + self.time != self.end:
            raise ValueError("time and end values don't match. You only need to specify one.")

        td_fields = [
            "time_delta", "time", "fastforward",
            "downtime_first", "downtime_interval", "downtime_length",
        ]
        # Check time fields are divisible by time_unit.
        for field in td_fields:
            td = getattr(self, field)
            if td is not None:
                convert_to_time_unit(td, self.time_unit)  # will throw if invalid

        if "workload" not in self.model_fields_set and self.replay:
            self.workload = "replay"  # default to replay if --replay is set
        if self.workload == "replay" and not self.replay:
            raise ValueError('--replay must be set when workload type is "replay"')
        elif self.workload != "replay" and self.replay:
            raise ValueError('workload must be either omitted or "replay" when --replay is set')

        if self.cooling:
            self.layout = "layout2"

        if 'weather' not in self.model_fields_set:
            self.weather = self.cooling and bool(self.replay)

        if self.jobsize_is_power_of is not None and self.jobsize_is_of_degree is not None:
            raise ValueError("jobsize_is_power_of and jobsize_is_of_degree are mutually exclusive")

        if self.plot and not self.output:
            raise ValueError("plot requires an output directory to be set")

        if self.live and not self.replay and self.time is None:
            raise ValueError("--time must be set, specifing how long we want to predict")

        if self.policy or self.backfill:
            try:
                module = importlib.import_module(f"raps.schedulers.{self.scheduler}")
            except ImportError as e:
                raise ValueError(f"Scheduler '{self.scheduler}' could not be imported") from e

        if self.policy:
            extended_policytypes = getattr(module, "ExtendedPolicyType", None)

            valid_policies = set(m.value for m in PolicyType)
            if extended_policytypes is not None:
                valid_policies |= {m.value for m in extended_policytypes}

            if self.policy not in valid_policies:
                raise ValueError(f"policy {self.policy} not implemented by {self.scheduler}. "
                                 f"Valid selections: {sorted(valid_policies)}")

        if self.backfill:
            extended_backfilltypes = getattr(module, "ExtendedBackfillType", None)

            valid_backfilltypes = set(m.value for m in BackfillType)
            if extended_backfilltypes is not None:
                valid_backfilltypes |= {m.value for m in extended_backfilltypes}

            if self.backfill not in valid_backfilltypes:
                raise ValueError(f"policy {self.backfill} not implemented by {self.scheduler}. "
                                 f"Valid selections: {sorted(valid_backfilltypes)}")

        return self

    @property
    @abc.abstractmethod
    def system_name(self) -> str:
        """
        Name of the system.
        Note, this is different than system, as system can be a file, or there can be multiple systems
        """
        pass

    @property
    @abc.abstractmethod
    def system_configs(self) -> list[SystemConfig]:
        """
        Return the SystemConfigs for the selected systems.
        Will be a single element array unless multiple `partitions` are selected.
        """
        pass

    def get_system_config_by_name(self, name: str) -> SystemConfig:
        for s in self.system_configs:
            if s.system_name == name:
                return s
        raise ValueError(f"Partition {name} isn't in SimConfig")

    def get_legacy_args(self):
        """
        Return as an argparse.Namespace object for backwards compatability
        """
        return argparse.Namespace(**self.get_legacy_args_dict())

    def get_legacy_args_dict(self):
        """
        Return as a dict object. This is for backwards compatibility with the rest of RAPS code so
        we can migrate to the new config gradually. The dict also has a "sim_config" key that
        contains the SimConfig object itself.
        """
        args_dict = self.model_dump(mode="json")
        args_dict['system'] = self.system_name
        # validate has been renamed to power_scope
        args_dict['validate'] = self.power_scope == "node"
        args_dict['downscale'] = self.downscale

        # Convert Path objects to str
        if self.output:
            args_dict['output'] = str(self.output)
        if self.replay:
            args_dict['replay'] = [str(p) for p in self.replay]
        if self.accounts_json:
            args_dict['accounts_json'] = str(self.accounts_json)

        args_dict["time"] = self.time_int
        args_dict["time_delta"] = self.time_delta_int
        args_dict["downtime_first"] = self.downtime_first_int
        args_dict["downtime_interval"] = self.downtime_interval_int
        args_dict["downtime_length"] = self.downtime_length_int
        args_dict['start'] = self.start.astimezone().isoformat() if self.start else None
        args_dict['end'] = self.end.astimezone().isoformat() if self.end else None
        args_dict.pop("fastforward")  # Remove fastforward from this to avoid confusion later

        args_dict['sim_config'] = self
        return args_dict

    def dump_yaml(self, exclude_unset=True):
        return yaml_dump(self.model_dump(mode="json", exclude_unset=exclude_unset))


class SingleSimConfig(SimConfig, abc.ABC):
    # Dynamic help string
    system: A[SystemConfig | str, Field(description=f"""
        Name of the system to simulate or a path to a yaml file containing the SystemConfig.

        You can also make modifications to the SystemConfig on the CLI using `--system.base`, e.g
        `--system.base frontier --system.cooling.fmu-path path/to/my.fmu`.

        Built-in systems: {', '.join(list_systems())}
    """)] = "frontier"

    @model_validator(mode="after")
    def _validate_system(self, info):
        self.system = resolve_system_reference(self.system, info)
        try:
            self._system_configs = [get_system_config(self.system)]
        except FileNotFoundError as e:
            raise ValueError(str(e))
        return self

    @property
    def system_name(self) -> str:
        return self.system_configs[0].system_name

    @property
    def system_configs(self) -> list[SystemConfig]:
        return self._system_configs


class MultiPartSimConfig(SimConfig):
    partitions: A[list[SystemConfig | str], Len(min_length=1)]
    """
    List of multiple systems/partitions to run. Can be names of preconfigured systems, or paths
    to custom SystemConfig yaml files.
    """

    @model_validator(mode="after")
    def _validate_partitions(self, info):
        self.partitions = [resolve_system_reference(p, info) for p in self.partitions]
        try:
            self._multi_partition_system_config = get_partition_configs(self.partitions)
        except FileNotFoundError as e:
            raise ValueError(str(e))
        return self

    @property
    def system_name(self) -> str:
        return self._multi_partition_system_config.system_name

    @property
    def system_configs(self) -> list[SystemConfig]:
        return self._multi_partition_system_config.partitions


SIM_SHORTCUTS = {
    "partitions": "x",
    "cooling": "c",
    "simulate-network": "net",
    "time": "t",
    "fastforward": "ff",
    "debug": "d",
    "numjobs": "n",
    "verbose": "v",
    "output": "o",
    "uncertainties": "u",
    "plot": "p",
    "replay": "f",
    "workload": "w",
}
