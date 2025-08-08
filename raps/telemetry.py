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
#import itertools
import json
import os.path


if __name__ == "__main__":
    #from raps.args import args,args_dict
    parser = argparse.ArgumentParser(description='Telemetry data validator')
    parser.add_argument('--jid', type=str, default='*', help='Replay job id')
    parser.add_argument('-f', '--replay', nargs='+', type=str,
                        help='Either: path/to/joblive path/to/jobprofile' + \
                             ' -or- filename.npz (overrides --workload option)')
    parser.add_argument('-p', '--plot', type=str, default=None, choices=['jobs','nodes'], help='Output plots')
    parser.add_argument("--is-results-file", action='store_true', default=False, help='Output plots')
    parser.add_argument("--gantt-nodes", default=False, action='store_true', required=False, help="Print Gannt with nodes required as line thickness (default false)")  # duplicate in workload!
    parser.add_argument('-t', '--time', type=str, default=None, help='Length of time to simulate, e.g., 123, 123s, 27m, 3h, 7d')
    parser.add_argument('--system', type=str, default='frontier', help='System config to use')
    choices = ['prescribed', 'poisson']
    parser.add_argument('--arrival', default=choices[0], type=str, choices=choices, help=f'Modify arrival distribution ({choices[1]}) or use the original submit times ({choices[0]})')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    args_dict = vars(args)

import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.progress import track

from raps.config import ConfigManager
from raps.job import Job, job_dict
import matplotlib.pyplot as plt
from raps.plotting import Plotter, plot_submit_times, plot_nodes_histogram, plot_jobs_gantt, plot_nodes_gantt, spaced_colors, plot_network_histogram
from raps.utils import next_arrival_byconfargs, create_casename, convert_to_seconds
#from raps.args import args, args_dict


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

    def save_snapshot(self, *, jobs: list, timestep_start:int, timestep_end:int, args:dict, filename: str):
        """Saves a snapshot of the jobs to a compressed file. """
        list_of_job_dicts = []
        for job in jobs:
            list_of_job_dicts.append(job.__dict__)
        np.savez_compressed(filename, jobs=list_of_job_dicts, timestep_start=timestep_start, timestep_end=timestep_end, args=args)

    def load_snapshot(self, snapshot: str, downscale=1) -> list:
        """Reads a snapshot from a compressed file and return 4 values: joblist, timestep_start, timestep_end and args.

        :param str snapshot: Filename
        :returns:
            - job list
            - timestep_start
            - timestep_end
            - args, which were used to generate the loaded snapshot
        """
        data = np.load(snapshot, allow_pickle=True, mmap_mode='r')
        jobs = []
        list_of_job_dicts = data['jobs'].tolist()
        for job_info in list_of_job_dicts:
            jobs.append(Job(job_info))
        if hasattr(data,'timestep_start'):
            timestep_start = int(data['timestep_start'])
        else:
            timestep_start = 0
        if hasattr(data,'timestep_end'):
            timestep_end = int(data['timestep_end'])
        else:
            timestep_end = np.inf
        if hasattr(data,'args'):
            args_from_file = data['args'].tolist()
        else:
            args_from_file = None

        return jobs, \
               timestep_start, \
               timestep_end, \
               args_from_file

    def load_csv_results(self, file):
        jobs = []
        time_start = 0
        time_end = 0
        for line in pd.read_csv(file,chunksize=1):
            job_info = job_dict(nodes_required=line.get('num_nodes').item(),  # Named like this somewhere in the csv history dumper
                                name=line.get('name').item(),
                                account=line.get('account').item(),
                                cpu_trace=None,
                                gpu_trace=None,
                                ntx_trace=None,
                                nrx_trace=None,
                                #end_state=line.get('end_state').item(),
                                end_state=None,
                                scheduled_nodes=json.loads(line.get('scheduled_nodes').item()),
                                id=line.get('id').item(),
                                #priority=line.get('priority').item(),
                                priority=None,
                                #partition=line.get('partition').item(),
                                partition=None,
                                submit_time=line.get('submit_time').item(),
                                start_time=line.get('start_time').item(),
                                end_time=line.get('end_time').item(),
                                #wall_time=line.get('wall_time').item(),
                                wall_time=line.get('end_time').item() - line.get('start_time').item(),
                                #trace_time=line.get('trace_time').item(),
                                trace_time=None,
                                #trace_start_time=line.get('trace_start_time').item(),
                                trace_start_time=None,
                                #trace_end_time=line.get('trace_end_time').item(),
                                trace_end_time=None,
                                #trace_missing_values=line.get('trace_missing_values').item(),
                                trace_missing_values=None
                                )
            job = Job(job_info)
            jobs.append(job)
        #if hasattr(data,'args'):
        #    args_from_file = data["args"].item()  # This should be empty  as csv contains no args.
        #else:
        #    args_from_file = None

        return jobs, time_start, time_end, None

    def load_data(self, files):
        """Load telemetry data using custom data loaders."""
        return self.dataloader.load_data(files, **self.kwargs)

    def load_data_from_df(self, *args, **kwargs):
        """Load telemetry data using custom data loaders."""
        return self.dataloader.load_data_from_df(*args, **kwargs)

    def load_data_from_csv(self, file, *args, **kwargs):
        jobs = []
        df = pd.read_csv(file,chunksize=1, header='infer')
        for d in df:
            #print(d['name'].astype(str))
            job_info = job_dict(nodes_required=None,
                                name=d['name'].astype(str).item(),
                                account=d['account'].astype(str).item(),
                                cpu_trace=None,
                                gpu_trace=None,
                                ntx_trace=None,
                                nrx_trace=None,
                                end_state=d['state'].astype(str).item(),
                                scheduled_nodes=d['scheduled_nodes'].item(),
                                id=d['id'].astype(int).item(),
                                priority=None,
                                partition=None,
                                submit_time=d['submit_time'].astype(int).item(),
                                time_limit=None,
                                start_time=d['start_time'].astype(int).item(),
                                end_time=d['end_time'].astype(int).item(),
                                wall_time=d['end_time'].astype(int).item() - d['start_time'].astype(int).item(),
                                trace_time=None,
                                trace_start_time=None,
                                trace_end_time=None,
                                trace_missing_values=None
                                )
            jobs.append(job_info)
        minstarttime = min([x['start_time'] for x in jobs])
        maxendtime = max([x['end_time'] for x in jobs])
        return jobs, minstarttime, maxendtime, None

    def node_index_to_name(self, index: int):
        """ Convert node index into a name"""
        return self.dataloader.node_index_to_name(index, config=self.config)

    def cdu_index_to_name(self, index: int):
        """ Convert cdu index into a name"""
        return self.dataloader.cdu_index_to_name(index, config=self.config)

    def cdu_pos(self, index: int) -> tuple[int, int]:
        """ Return (row, col) tuple for a cdu index """
        return self.dataloader.cdu_pos(index, config=self.config)

    def load_jobs_times_args_from_files(self,*,files, args, downscale=1):
        """ Load all files as combined jobs """
        # Read telemetry data (either npz file or via custom data loader)
        # TODO: Merge args? See main.py:79
        timestep_end = 0
        timestep_start = sys.maxsize
        jobs = []
        trigger_custom_dataloader = False
        for i,file in enumerate(files):
            file = os.path.normpath(file.lstrip('"').rstrip('"'))
            if hasattr(args,'is_results_file') and args.is_results_file:
                if file.endswith(".csv"):
                    jobs, timestep_start, timestep, _ = self.load_csv_results(file)

            elif file.endswith(".npz"):  # Replay .npz file
                print(f"Loading {file}...")
                jobs_from_file, timestep_start_from_file, timestep_end_from_file, args_from_file = self.load_snapshot(file)
                if args_from_file is not None:
                    print("File was generated with:" +\
                          f"\n--system {args_from_file.system} " +\
                          f"-ff {args_from_file.fastforward} " +\
                          f"-t {args_from_file.time}\n" +\
                          f"All Args:\n{args_from_file}" +\
                          "To use these set them from the commandline!"
                          )
                else:
                    print("No generation arguments extracted from input file!")
                    # Args are usually extracted to tell the users how to reporduce results.
                    # They are not processed and re-set to said arguments automatily
                jobs.extend(jobs_from_file)
                timestep_start = min(timestep_start,timestep_start_from_file)
                timestep_end = max(timestep_end, timestep_end_from_file)

                if hasattr(args,'scale') and args.scale:
                    for job in tqdm(jobs, desc=f"Scaling jobs to {args.scale} nodes"):
                        job['nodes_required'] = random.randint(1, args.scale)
                        job['scheduled_nodes'] = None  # Setting to None triggers scheduler to assign nodes

                if hasattr(args,'arrival') and args.arrival == 'poisson':
                    print("available nodes:", config['AVAILABLE_NODES'])
                    for job in tqdm(jobs, desc="Rescheduling jobs"):
                        job['scheduled_nodes'] = None
                        job['submit_time'] = next_arrival_byconfargs(config,args)
            else:
                trigger_custom_dataloader = True
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
            try:
                jobs, timestep_start_from_data, timestep_end_from_data = self.load_data(args.replay)
            except AssertionError:
                raise ValueError("Forgot --is-results-file ?")
            timestep_start = min(timestep_start, timestep_start_from_data)
            timestep_end = max(timestep_end, timestep_end_from_data)
            self.save_snapshot(jobs=jobs,
                               timestep_start=timestep_start,
                               timestep_end=timestep_end,
                               args=args, filename=self.dirname)
        if args.time:
            timestep_end = timestep_start + convert_to_seconds(args.time)
        elif not timestep_end:
            timestep_end = int(max(job.wall_time + job.start_time for job in jobs)) + 1

        return jobs, timestep_start, timestep_end, args


if __name__ == "__main__":
    config = ConfigManager(system_name=args.system).get_config()
    args_dict['config'] = config
    td = Telemetry(**args_dict)
    if args.replay:
        jobs, timestep_start, timestep_end, _ = td.load_jobs_times_args_from_files(files=args.replay,args=args)

    else:
        parser.print_help()

    timesteps = timestep_end - timestep_start

    dt_list = []
    wt_list = []
    nr_list = []
    submit_times = []
    end_times = []
    last = 0
    for job in jobs:
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
    dt_list = [item for item in dt_list if item is not None]
    nr_list = [item for item in nr_list if item is not None]
    wt_list = [item for item in wt_list if item is not None]

    print(f'Number of jobs: {len(jobs)}')
    print(f'Simulation will run for {timesteps} seconds')
    if dt_list:
        print(f'Average job arrival time is: {np.mean(dt_list):.2f}s')
    if wt_list:
        print(f'Average wall time is: {np.mean(wt_list):.2f}s')
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
        if hasattr(job_vec,'ntx_trace'):
            ntx = np.array(job_vec.get('ntx_trace', []))
            if ntx.size > 0 and not np.all(np.isnan(ntx)):
                ntx_means.append(np.nanmean(ntx))
        if hasattr(job_vec,'nrx_trace'):
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
        fig,ax = plt.subplots()
    if args.plot == "jobs":
        plot_jobs_gantt(ax=ax,jobs=jobs, bars_are_node_sized=args.gantt_nodes)
        ax.invert_yaxis()
    elif args.plot == "nodes":
        plot_nodes_gantt(ax=ax,jobs=jobs)
    elif args.plot == "network":
        if ntx_means and nrx_means:
            # combine into total per‐job traffic
            net_means = [tx + rx for tx, rx in zip(ntx_means, nrx_means)]
            plot_network_histogram(ax=ax,data=net_means)
    plt.show()
