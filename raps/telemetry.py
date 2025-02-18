"""
This module provides functionality for handling telemetry data, including encryption,
index conversion, and job data parsing. It supports reading and saving snapshots,
parsing parquet files, and generating job state information.

The module defines a `Telemetry` class for managing telemetry data and several
helper functions for data encryption and conversion between node name and index formats.
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telemetry data validator')
    parser.add_argument('--jid', type=str, default='*', help='Replay job id')
    parser.add_argument('-f', '--replay', nargs='+', type=str,
                        help='Either: path/to/joblive path/to/jobprofile' + \
                             ' -or- filename.npz (overrides --workload option)')
    parser.add_argument('-p', '--plot', action='store_true', help='Output plots')
    parser.add_argument('--system', type=str, default='frontier', help='System config to use')
    choices = ['prescribed', 'poisson']
    parser.add_argument('--arrival', default=choices[0], type=str, choices=choices, help=f'Modify arrival distribution ({choices[1]}) or use the original submit times ({choices[0]})')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

import importlib
import numpy as np
from tqdm import tqdm

from .config import ConfigManager
from .job import Job
from .account import Accounts
from .plotting import plot_submit_times, plot_nodes_histogram
from .utils import next_arrival


class Telemetry:
    """A class for handling telemetry data, including reading/parsing job data, and loading/saving snapshots."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.system = kwargs.get('system')
        self.config = kwargs.get('config')
        try:
            self.dataloader = importlib.import_module(f".dataloaders.{self.system}", package=__package__)
        except:
            print("WARNING: Failed to load dataloader")

    def save_snapshot(self, jobs: list, filename: str):
        """Saves a snapshot of the jobs to a compressed file. """
        np.savez_compressed(filename, jobs=jobs)

    def load_snapshot(self, snapshot: str) -> list:
        """Reads a snapshot from a compressed file and returns the jobs."""
        jobs = np.load(snapshot, allow_pickle=True, mmap_mode='r')
        return jobs['jobs'].tolist()

    def load_data(self, files):
        """Load telemetry data using custom data loaders."""
        return self.dataloader.load_data(files, **self.kwargs)

    def load_data_from_df(self, *args, **kwargs):
        """Load telemetry data using custom data loaders."""
        return self.dataloader.load_data_from_df(*args, **kwargs)

    def node_index_to_name(self, index: int):
        """ Convert node index into a name"""
        return self.dataloader.node_index_to_name(index, config=self.config)

    def cdu_index_to_name(self, index: int):
        """ Convert cdu index into a name"""
        return self.dataloader.cdu_index_to_name(index, config=self.config)

    def cdu_pos(self, index: int) -> tuple[int, int]:
        """ Return (row, col) tuple for a cdu index """
        return self.dataloader.cdu_pos(index, config=self.config)


if __name__ == "__main__":

    args_dict = vars(args)
    config = ConfigManager(system_name=args.system).get_config()
    args_dict['config'] = config
    td = Telemetry(**args_dict)

    if args.replay[0].endswith(".npz"):
        print(f"Loading {args.replay[0]}...")
        jobs = td.load_snapshot(args.replay[0])
        if args.arrival == "poisson":
            for job in tqdm(jobs, desc="Updating requested_nodes"):
                job['requested_nodes'] = None
                job['submit_time'] = next_arrival(1 / config['JOB_ARRIVAL_TIME'])
    else:
        jobs = td.load_data(args.replay)

    timesteps = int(max(job['wall_time'] + job['submit_time'] for job in jobs))

    dt_list = []
    wt_list = []
    nr_list = []
    submit_times = []
    last = 0
    for job_vector in jobs:
        job = Job(job_vector, 0)  # current_time is never used in Job()
        wt_list.append(job.wall_time)
        nr_list.append(job.nodes_required)
        submit_times.append(job.submit_time)
        if job.submit_time > 0:
            dt = job.submit_time - last
            dt_list.append(dt)
            last = job.submit_time
        if args.verbose:
            print(job)

    print(f'Simulation will run for {timesteps} seconds')
    print(f'Average job arrival time is: {np.mean(dt_list):.2f}s')
    print(f'Average wall time is: {np.mean(wt_list):.2f}s')
    print(f'Nodes required (avg): {np.mean(nr_list):.2f}')
    print(f'Nodes required (max): {np.max(nr_list)}')
    print(f'Nodes required (std): {np.std(nr_list):.2f}')

    if args.plot:
        plot_nodes_histogram(nr_list)
        plot_submit_times(submit_times, nr_list)
