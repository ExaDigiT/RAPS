"""Workloads package init."""

import math
import numpy as np

from raps.utils import WorkloadData, SubParsers
from raps.utils import pydantic_add_args
from raps.sim_config import SingleSimConfig

from .basic import BasicWorkload
from .constants import JOB_NAMES, ACCT_NAMES, MAX_PRIORITY
from .distribution import DistributionWorkload
from .live import continuous_job_generation, run_workload
from .multitenant import MultitenantWorkload
from .utils import plot_job_hist


class BaseWorkload:
    """Base class with common workload logic."""

    def __init__(self, args, *configs):
        self.partitions = [c['system_name'] for c in configs]
        self.config_map = {c['system_name']: c for c in configs}
        self.args = args

    def generate_jobs(self):
        jobs = getattr(self, self.args.workload)(args=self.args)
        timestep_end = int(math.ceil(max([job.end_time for job in jobs])))
        return WorkloadData(
            jobs=jobs,
            telemetry_start=0,
            telemetry_end=timestep_end,
            start_date=self.args.start,
        )

    def compute_traces(self,
                       cpu_util: float,
                       gpu_util: float,
                       expected_run_time: int,
                       trace_quanta: int
                       ) -> tuple[np.ndarray, np.ndarray]:
        """ Compute CPU and GPU traces based on mean CPU & GPU utilizations and wall time. """
        cpu_trace = cpu_util * np.ones(int(expected_run_time) // trace_quanta)
        gpu_trace = gpu_util * np.ones(int(expected_run_time) // trace_quanta)
        return (cpu_trace, gpu_trace)
        
class Workload(
    BaseWorkload,
    DistributionWorkload,
    BasicWorkload,
    MultitenantWorkload
):
    """Final workload class with all workload types."""
    pass

__all__ = [
    "Workload",
    "JOB_NAMES", "ACCT_NAMES", "MAX_PRIORITY",
]


def run_workload_add_parser(subparsers: SubParsers):
    from raps.sim_config import SIM_SHORTCUTS
    # TODO: Separate the arguments for this command
    parser = subparsers.add_parser("workload", description="""
        Saves workload as a snapshot.
    """)
    parser.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    model_validate = pydantic_add_args(parser, SingleSimConfig, model_config={
        "cli_shortcuts": SIM_SHORTCUTS,
    })
    parser.set_defaults(impl=lambda args: run_workload(model_validate(args, {})))
