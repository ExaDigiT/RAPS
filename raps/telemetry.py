"""
This module provides functionality for handling telemetry data, including encryption,
index conversion, and job data parsing. It supports reading and saving snapshots,
parsing parquet files, and generating job state information.

The module defines a `Telemetry` class for managing telemetry data and several
helper functions for data encryption and conversion between node name and index formats.
"""
from typing import Literal
import sys
import random
from pathlib import Path
# import json
from typing import Optional
from types import ModuleType
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, model_validator
# from rich.progress import track

from raps.sim_config import SimConfig
from raps.system_config import get_system_config
from raps.job import Job, job_dict
import matplotlib.pyplot as plt
from raps.plotting import (
    plot_jobs_gantt,
    plot_nodes_gantt,
    plot_network_histogram
)
from raps.utils import (
    next_arrival_byconfargs, convert_to_time_unit, pydantic_add_args, SubParsers, ExpandedPath,
    WorkloadResult,
)


# TODO: should reuse this model in SimConfig
class TelemetryArgs(BaseModel):
    jid: str = '*'
    """ Replay job id """
    replay: list[ExpandedPath] | None = None
    """ path/to/joblive path/to/jobprofile  -or- filename.npz (overrides --workload option) """
    plot: list[Literal["jobs", "nodes"]] | None = None
    """ Output plots """
    gantt_nodes: bool = False
    """ Print Gannt with nodes required as line thickness (default false) """
    time: str | None = None
    """ Length of time to simulate, e.g., 123, 123s, 27m, 3h, 7d """
    system: str = 'frontier'
    """ System config to use """
    arrival: Literal['prescribed', 'poisson'] = "prescribed"
    """ Modify arrival distribution ({choices[1]}) or use the original submit times """
    verbose: bool = False
    output: str | None = None
    """ Store output in --output <arg> file. """
    live: bool = False
    """ Grab data from live system. """

    @model_validator(mode="after")
    def _validate_after(self):
        if not self.live and not self.replay:
            raise ValueError("Either --live or --replay is required")
        return self


shortcuts = {
    "replay": "f",
    "plot": "p",
    "time": "t",
    "verbose": "v",
    "output": "o",
}


class Telemetry:
    """A class for handling telemetry data, including reading/parsing job data, and loading/saving snapshots."""
    dataloader: Optional[ModuleType]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.system = kwargs.get('system')
        self.config = kwargs.get('config')

        try:
            self.dataloader = importlib.import_module(f"raps.dataloaders.{self.system}", package=__package__)
        except ImportError as e:
            print(f"WARNING: Failed to load dataloader: {e}")
            self.dataloader = None

    def save_snapshot(self, *, dest: str, result: WorkloadResult, args: SimConfig|TelemetryArgs):
        """Saves a snapshot of the jobs to a compressed file. """
        np.savez_compressed(dest,
            jobs=[vars(j) for j in result.jobs],
            telemetry_start=result.telemetry_start,
            telemetry_end=result.telemetry_end,
            start_date=result.start_date,
            args=args,
        )

    def load_snapshot(self, snapshot: str | Path) -> tuple[WorkloadResult, SimConfig|TelemetryArgs]:
        """Reads a snapshot from a compressed file

        :param str snapshot: Filename
        :returns:
            - job list
            - timestep_start
            - timestep_end
            - args, which were used to generate the loaded snapshot
        """
        data = np.load(snapshot, allow_pickle=True, mmap_mode='r')
        jobs = [Job(j) for j in data['jobs']]
        telemetry_start = data['telemetry_start'].item()
        telemetry_end = data['telemetry_end'].item()
        start_date = data['start_date'].item()
        args = data['args'].item()

        result = WorkloadResult(
            jobs=jobs,
            telemetry_start=telemetry_start, telemetry_end=telemetry_end,
            start_date=start_date,
        )

        return result, args

    def load_data(self, files):
        """Load telemetry data using custom data loaders."""
        assert self.dataloader
        return self.dataloader.load_data(files, **self.kwargs)

    def load_live_data(self):
        """Load telemetry data using custom data loaders."""
        assert self.dataloader
        return self.dataloader.load_live_data(**self.kwargs)

    def node_index_to_name(self, index: int):
        """ Convert node index into a name"""
        assert self.dataloader
        return self.dataloader.node_index_to_name(index, config=self.config)

    def cdu_index_to_name(self, index: int):
        """ Convert cdu index into a name"""
        assert self.dataloader
        return self.dataloader.cdu_index_to_name(index, config=self.config)

    def cdu_pos(self, index: int) -> tuple[int, int]:
        """ Return (row, col) tuple for a cdu index """
        assert self.dataloader
        return self.dataloader.cdu_pos(index, config=self.config)

    def load_from_live_system(self):
        result = self.load_live_data()
        return result

    def load_from_files(self, *, files, args, config):
        """ Load all files as combined jobs """
        assert len(files) >= 1
        files = [Path(f) for f in files]

        if str(files[0]).endswith(".npz"):
            file = files[0]
            print(f"Loading {file}")
            result, args_from_file = self.load_snapshot(file)
            print(f"File was generated with: --system {args_from_file.system}")

            # TODO: should move this logic into a separate method and out of the individual dataloaders
            if hasattr(args, 'scale') and args.scale:
                for job in tqdm(result.jobs, desc=f"Scaling jobs to {args.scale} nodes"):
                    job.nodes_required = random.randint(1, args.scale)
                    job.scheduled_nodes = None  # Setting to None triggers scheduler to assign nodes

            if hasattr(args, 'arrival') and args.arrival == 'poisson':
                print("available nodes:", config['AVAILABLE_NODES'])
                for job in tqdm(result.jobs, desc="Rescheduling jobs"):
                    job.scheduled_nodes = None
                    job.submit_time = next_arrival_byconfargs(config, args)
                    job.start_time = None
                    job.end_time = None
        else: # custom data loader
            result = self.load_data(args.replay)
        if args.time:
            result.telemetry_end = result.telemetry_start + convert_to_time_unit(args.time)
        return result


def run_telemetry_add_parser(subparsers: SubParsers):
    parser = subparsers.add_parser("telemetry", description="""
        Telemetry data validator
    """)
    model_validate = pydantic_add_args(parser, TelemetryArgs, {
        "cli_shortcuts": shortcuts,
    })
    parser.set_defaults(impl=lambda args: run_telemetry(model_validate(args, {})))


def run_telemetry(args: TelemetryArgs):
    args_dict = args.model_dump()
    config = get_system_config(args.system).get_legacy()
    args_dict['config'] = config
    td = Telemetry(**args_dict)

    if args.live and not args.replay:
        result = td.load_from_live_system()
    else:
        result = td.load_from_files(files=args.replay, args=args, config=config)
    jobs = result.jobs
    timestep_start = result.telemetry_start
    timestep_end = result.telemetry_end

    if args.output:
        td.save_snapshot(dest = args.output, result = result, args = args)

    timesteps = timestep_end - timestep_start

    dt_list = []  # arrival time ???
    tl_list = []  # time limit
    ert_list = []  # expected run time
    nr_list = []  # nodes required
    submit_times = []
    end_times = []
    last = 0
    for job in jobs:
        tl_list.append(job.time_limit)
        ert_list.append(job.expected_run_time)
        nr_list.append(job.nodes_required)
        submit_times.append(job.submit_time)
        end_times.append(job.submit_time + job.time_limit)
        if job.submit_time > 0:
            dt = job.submit_time - last
            dt_list.append(dt)
            last = job.submit_time
        if args.verbose:
            print(job)
    dt_list = [item for item in dt_list if item is not None]
    nr_list = [item for item in nr_list if item is not None]
    tl_list = [item for item in tl_list if item is not None]
    ert_list = [item for item in ert_list if item is not None]

    print(f'Number of jobs: {len(jobs)}')
    print(f'Simulation will run for {timesteps} seconds')
    if dt_list:
        print(f'Average job arrival time is: {np.mean(dt_list):.2f}s')
    if tl_list:
        print(f'Average time limit is: {np.mean(tl_list):.2f}s')
    if ert_list:
        print(f'Average expected runtime is: {np.mean(ert_list):.2f}s')

    if nr_list:
        print(f'Nodes required (avg): {np.mean(nr_list):.2f}')
        print(f'Nodes required (max): {np.max(nr_list)}')
        print(f'Nodes required (std): {np.std(nr_list):.2f}')

    # ——— compute avg network traces ———
    ntx_means = []
    nrx_means = []
    for job in jobs:
        job_vec = job.__dict__
        # only if there’s at least one valid sample
        if hasattr(job_vec, 'ntx_trace'):
            ntx = np.array(job_vec.get('ntx_trace', []))
            if ntx.size > 0 and not np.all(np.isnan(ntx)):
                ntx_means.append(np.nanmean(ntx))
        if hasattr(job_vec, 'nrx_trace'):
            nrx = np.array(job_vec.get('nrx_trace', []))
            if nrx.size > 0 and not np.all(np.isnan(nrx)):
                nrx_means.append(np.nanmean(nrx))

    if ntx_means:
        print(f'Average ntx_trace per job: {np.mean(ntx_means):.2f}')
    else:
        print('No valid ntx_trace data found.')

    if nrx_means:
        print(f'Average nrx_trace per job: {np.mean(nrx_means):.2f}')
    else:
        print('No valid nrx_trace data found.')

    if args.plot:
        fig, ax = plt.subplots()
    if args.plot == "jobs":
        plot_jobs_gantt(ax=ax, jobs=jobs, bars_are_node_sized=args.gantt_nodes)
        ax.invert_yaxis()
    elif args.plot == "nodes":
        plot_nodes_gantt(ax=ax, jobs=jobs)
    elif args.plot == "network":
        if ntx_means and nrx_means:
            # combine into total per‐job traffic
            net_means = [tx + rx for tx, rx in zip(ntx_means, nrx_means)]
            plot_network_histogram(ax=ax, data=net_means)
    if args.output is not None:
        if args.output == "":
            filename = f"{args.output}.svg"
        else:
            filename = args.output
        plt.savefig(f'{filename}')
        print(f"Saved to: {filename}")
    else:
        plt.show()
