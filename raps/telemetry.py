"""
This module provides functionality for handling telemetry data, including encryption,
index conversion, and job data parsing. It supports reading and saving snapshots,
parsing parquet files, and generating job state information.

The module defines a `Telemetry` class for managing telemetry data and several
helper functions for data encryption and conversion between node name and index formats.
"""
import re
import sys
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telemetry data validator')
    parser.add_argument('--jid', type=str, default='*', help='Replay job id')
    parser.add_argument('-f', '--replay', nargs='+', type=str,
                        help='Either: path/to/joblive path/to/jobprofile' + \
                             ' -or- filename.npz (overrides --workload option)')
    parser.add_argument('-p', '--plot', action='store_true', help='Output plots')
    parser.add_argument('-t', '--time', type=str, default=None, help='Length of time to simulate, e.g., 123, 123s, 27m, 3h, 7d')
    parser.add_argument('--system', type=str, default='frontier', help='System config to use')
    choices = ['prescribed', 'poisson']
    parser.add_argument('--arrival', default=choices[0], type=str, choices=choices, help=f'Modify arrival distribution ({choices[1]}) or use the original submit times ({choices[0]})')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    args_dict = vars(args)

import importlib
import numpy as np
from tqdm import tqdm

from raps.config import ConfigManager
from raps.job import Job
#from raps.account import Accounts
from raps.plotting import plot_submit_times, plot_nodes_histogram, plot_job_gantt
from raps.utils import next_arrival_byconfargs, create_casename, convert_to_seconds


class Telemetry:
    """A class for handling telemetry data, including reading/parsing job data, and loading/saving snapshots."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.system = kwargs.get('system')
        self.config = kwargs.get('config')
        self.dirname = create_casename()
        try:
            self.dataloader = importlib.import_module(f"raps.dataloaders.{self.system}", package=__package__)
        except:
            print("WARNING: Failed to load dataloader")

    def save_snapshot(self,*, jobs: list, timestep_start, timestep_end, args, filename: str):
        """Saves a snapshot of the jobs to a compressed file. """
        np.savez_compressed(filename, jobs=jobs, timestep_start=timestep_start, timestep_end=timestep_end, args=args)

    def load_snapshot(self, snapshot: str) -> list:
        """Reads a snapshot from a compressed file and return 4 values: joblist, timestep_start, timestep_end and args.

        :param str snapshot: Filename
        :returns:
            - job list
            - timestep_start
            - timestep_end
            - args, which were used to generate the loaded snapshot
        """
        data = np.load(snapshot, allow_pickle=True, mmap_mode='r')
        return data['jobs'].tolist(), \
               int(data['timestep_start']), \
               int(data['timestep_end']), \
               data['args'].tolist()

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

    def load_jobs_times_args_from_files(self,*,files, args):
        """ Load all files as combined jobs """
        # Read telemetry data (either npz file or via custom data loader)
        # TODO: Merge args? See main.py:79
        timestep_end = 0
        timestep_start = sys.maxsize
        jobs = []
        trigger_custom_dataloader = False
        for i,file in enumerate(files):
            if file.endswith(".npz"):  # Replay .npz file
                print(f"Loading {file}...")
                jobs_from_file, timestep_start_from_file, timestep_end_from_file, args_from_file = self.load_snapshot(file)
                if not hasattr(args_from_file,'fastforward') or args_from_file.fastforward is None:
                    args_from_file.fastforward = 0
                print("File was generated with:" +\
                      f"\n--system {args_from_file.system} " +\
                      f"-ff {args_from_file.fastforward} " +\
                      f"-t {args_from_file.time}\n" +\
                      f"All Args:\n{args_from_file}" +\
                      "To use these set them from the commandline!"
                      )
                jobs.extend(jobs_from_file)
                timestep_start = min(timestep_start,timestep_start_from_file)
                timestep_end = max(timestep_end, timestep_end_from_file)

                if hasattr(args,'scale') and args.scale:
                    for job in tqdm(jobs, desc=f"Scaling jobs to {args.scale} nodes"):
                        job['nodes_required'] = random.randint(1, args.scale)
                        job['requested_nodes'] = None  # Setting to None triggers scheduler to assign nodes

                if hasattr(args,'policy') and args.policy == 'poisson':
                    print("available nodes:", config['AVAILABLE_NODES'])
                    for job in tqdm(jobs, desc="Rescheduling jobs"):
                        job['requested_nodes'] = None
                        job['submit_time'] = next_arrival_byconfargs(config,args)
            elif i == 0:
                trigger_custom_dataloader = True
                break
            else:
                print("Multiple files given as input.")
                break

        if trigger_custom_dataloader:  # custom data loader
            # Try to extract date from given name to use as case directory
            matched_date = re.search(r"\d{4}-\d{2}-\d{2}", args.replay[0])
            if matched_date:
                extracted_date = matched_date.group(0)
                self.dirname = "sim=" + extracted_date
            else:
                extracted_date = "Date not found"
                self.dirname = create_casename()

            print(*args.replay)
            jobs, timestep_start_from_data, timestep_end_from_data = self.load_data(args.replay)
            timestep_start = min(timestep_start, timestep_start_from_data)
            timestep_end = max(timestep_end, timestep_end_from_data)
            self.save_snapshot(jobs=jobs,
                               timestep_start=timestep_start,
                               timestep_end=timestep_end,
                               args=args, filename=self.dirname)
        if args.time:
            timestep_end = timestep_start + convert_to_seconds(args.time)
        elif not timestep_end:
            timestep_end = int(max(job['wall_time'] + job['start_time'] for job in jobs)) + 1

        return jobs, timestep_start, timestep_end, args


if __name__ == "__main__":
    config = ConfigManager(system_name=args.system).get_config()
    args_dict['config'] = config
    td = Telemetry(**args_dict)
    jobs, timestep_start, timestep_end, _ = td.load_jobs_times_args_from_files(files=args.replay,args=args)

    timesteps = timestep_end - timestep_start

    dt_list = []
    wt_list = []
    nr_list = []
    submit_times = []
    end_times = []
    last = 0
    for job_vector in jobs:
        job = Job(job_vector)
        wt_list.append(job.wall_time)
        nr_list.append(job.nodes_required)
        submit_times.append(job.submit_time)
        end_times.append(job.submit_time + job.wall_time)
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
        #plot_nodes_histogram(nr_list)
        #plot_submit_times(submit_times, nr_list)
        plot_job_gantt(submit_times, end_times, nr_list)
