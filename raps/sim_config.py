import argparse
from functools import cached_property
from datetime import timedelta
from typing import Literal
from raps.schedulers.default import PolicyType, BackfillType
from raps.utils import (
    parse_time_unit, convert_to_time_unit, infer_time_unit, ExpandedPath, parse_td,
)
from raps.system_config import SystemConfig, get_partition_configs
from pydantic import BaseModel, model_validator
import importlib

Distribution = Literal['uniform', 'weibull', 'normal']


class SimConfig(BaseModel):
    system: str | None = None
    """ System config to use """
    partitions: list[str] = []
    """ List of multiple system configurations for a multi-partition run. Can contain wildcards """

    cooling: bool = False
    """ Include the FMU cooling model """
    simulate_network: bool = False
    """ Include network model """

    # Simulation runtime options
    fastforward: int | None = None
    """
    Fast-forward by time amount (unit specified by `time_unit`, default seconds).
    Can pass a string like 15s, 1m, 1h
    """
    time: int | None = None
    """
    Length of time to simulate (unit specified by `time_unit`, default seconds).
    Can pass a string like 123, 27m, 3h, 7d
    """
    time_delta: int = 1
    """
    Step size (unit specified by `time_unit`, default seconds).
    Can pass a string like 15s, 1m, 1h, 1ms
    """
    time_unit: timedelta = timedelta(seconds=1)
    """
    Units all time delta ints are measured in (default seconds)
    """

    @cached_property
    def downscale(self) -> int:
        return int(timedelta(seconds=1) / self.time_unit)

    start: str = "2021-05-21T13:00:00-04:00"
    """ ISO8601 start of simulation """
    end: str = "2021-05-21T14:00:00-04:00"
    """ ISO8601 end of simulation """

    numjobs: int = 100
    """ Number of jobs to schedule """

    uncertainties: bool = False
    """ Use float-with-uncertainties (much slower) """

    seed: int | None = None
    """ Set RNG seed for deterministic simulation """
    output: ExpandedPath | None = None
    """ Output power, cooling, and loss models for later analysis. Argument specifies name. """

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

    replay: list[ExpandedPath] | None = None
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
    workload: Literal['random', 'benchmark', 'peak', 'idle', 'synthetic', 'multitenant'] | None = None

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
    accounts_json: ExpandedPath | None = None
    """ Path to accounts JSON file from previous run """

    # Downtime
    downtime_first: int | None = None
    """
    First downtime (unit specified by `time_unit`, default seconds).
    Can pass a string like 27m, 3h, 7d
    """
    downtime_interval: str | None = None
    """
    Interval between downtimes (unit specified by `time_unit`, default seconds).
    Can pass a string like 123, 27m, 3h, 7d
    """
    downtime_length: str | None = None
    """
    Downtime length (unit specified by `time_unit`, default seconds).
    Can pass a string like 123, 27m, 3h, 7d
    """

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

        time_fields = [
            "time_delta", "time", "fastforward",
            "downtime_first", "downtime_interval", "downtime_length",
        ]

        if data.get('time_unit') is not None:
            time_unit = parse_time_unit(data['time_unit'])
            input_time_unit = time_unit
        else:
            time_unit = min(
                [infer_time_unit(data[f]) for f in time_fields if data.get(f)],
                default=timedelta(seconds=1)
            )
            # When "inferring" time unit interpret raw numbers as seconds.
            # E.g. `-t 10 --time-delta 1ds` should be `-t 10s --time-delta 1ds`
            input_time_unit = timedelta(seconds=1)

        data['time_unit'] = time_unit
        for field in time_fields:
            if data.get(field) is not None:
                td = parse_td(data[field], input_time_unit)
                data[field] = convert_to_time_unit(td, time_unit)

        return data

    @model_validator(mode="after")
    def _validate_after(self):
        # This is called after Pydantic has parsed everything into the model
        if self.system and self.partitions:
            raise ValueError("system and partitions are mutually exclusive")
        elif not self.system and not self.partitions:
            self.system = "frontier"

        if not self.replay and not self.workload:
            self.workload = "random"

        if self.cooling:
            self.layout = "layout2"

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
    def system_name(self) -> str:
        """
        Name of the system.
        Note, this is different than system, as system can be a file or None if partition is set.
        """
        return self._multi_partition_system_config.system_name

    @property
    def system_configs(self) -> list[SystemConfig]:
        """
        Return the SystemConfigs for the selected systems.
        Will be a single element array unless multiple `partitions` are selected.
        """
        return self._multi_partition_system_config.partitions

    @cached_property
    def _multi_partition_system_config(self):
        return get_partition_configs(self.partitions if self.partitions else [self.system])

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
        # validate has been renamed to power_scope
        args_dict['validate'] = args_dict["power_scope"] == "node"
        args_dict['downscale'] = self.downscale

        # Convert Path objects to str
        if args_dict['output']:
            args_dict['output'] = str(args_dict['output'])
        if args_dict['replay']:
            args_dict['replay'] = [str(p) for p in args_dict['replay']]
        if args_dict['accounts_json']:
            args_dict['accounts_json'] = str(args_dict['accounts_json'])

        args_dict['sim_config'] = self
        return args_dict
