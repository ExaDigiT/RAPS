import argparse
import sys
import yaml
from datetime import timedelta
from pathlib import Path
from typing import Literal
from raps.schedulers.default import PolicyType, BackfillType

from raps.utils import (
    parse_time_unit, convert_to_time_unit, infer_time_unit, ExpandedPath,
    pydantic_add_args, yaml_dump, parse_td,
)

from pydantic import BaseModel, model_validator, computed_field
from pydantic_settings import SettingsConfigDict

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
    time_unit: timedelta
    """
    Units all time delta ints are measured in (default seconds)
    """

    @computed_field
    @property
    def downscale(self) -> int:
        return int(timedelta(seconds=1) / self.time_unit)

    start: str = "2021-05-21T13:00"
    """ ISO8601 start of simulation """
    end: str = "2021-05-21T14:00"
    """ ISO8601 end of simulation """

    numjobs: int = 100
    """ Number of jobs to schedule """

    uncertainties: bool = False
    """ Use float-with-uncertainties (much slower) """

    seed: bool = False
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
        "default", "scheduleflow", "nrel", "anl", "flux", "experimental", "multitenant",
    ] = "default"
    """ Scheduler name """
    policy: PolicyType | None = None
    """ Schedule policy """
    backfill: BackfillType | None = None
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

    # Reinforcment Learning
    episode_length: int = 1000
    """ Number of timesteps per RL episode (default 1000) """

    @model_validator(mode="before")
    def _parse_times(cls, data):
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
    def _validate(self):
        if self.system and self.partitions:
            raise ValueError("system and partitions are mutually exclusive")
        elif not self.system and not self.partitions:
            self.system = "frontier"

        if not self.replay and not self.workload:
            self.workload = "random"

        if self.jobsize_is_power_of is not None and self.jobsize_is_of_degree is not None:
            raise ValueError("jobsize_is_power_of and jobsize_is_of_degree are mutually exclusive")

        return self

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

        # Convert Path objects to str
        if args_dict['output']:
            args_dict['output'] = str(args_dict['output'])
        if args_dict['replay']:
            args_dict['replay'] = [str(p) for p in args_dict['replay']]
        if args_dict['accounts_json']:
            args_dict['accounts_json'] = str(args_dict['accounts_json'])

        args_dict['sim_config'] = self
        return args_dict


def parse_args(cli_args=None) -> SimConfig:
    parser = argparse.ArgumentParser(
        description="Resource Allocator & Power Simulator (RAPS)",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file", nargs="?", default=None,
        help=(
            'YAML sim config file, can be used to configure an experiment instead of using CLI ' +
            'flags. Pass "-" to read from stdin.'
        )
    )

    model_validate_args = pydantic_add_args(parser, SimConfig, model_config=SettingsConfigDict(
        cli_implicit_flags=True,
        cli_kebab_case=True,
        cli_shortcuts={
            "partitions": "x",
            "cooling": "c",
            "simulate-network": "net",
            "fastforward": "ff",
            "time": "t",
            "debug": "d",
            "numjobs": "n",
            "verbose": "v",
            "output": "o",
            "uncertainties": "u",
            "plot": "p",
            "replay": "f",
            "workload": "w",
        },
    ))

    args = parser.parse_args(cli_args)
    if args.config_file == "-":
        config_file_data = yaml.safe_load(sys.stdin.read())
    elif args.config_file:
        config_file_data = yaml.safe_load(Path(args.config_file).read_text())
    else:
        config_file_data = {}

    return model_validate_args(args, config_file_data)


sim_config = parse_args()
args = sim_config.get_legacy_args()
args_dict = sim_config.get_legacy_args_dict()

if __name__ == "__main__":
    print(yaml_dump(sim_config.model_dump(mode="json")))
