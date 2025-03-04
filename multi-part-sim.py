from raps.helpers import check_python_version
check_python_version()

import glob
import os
import random
import sys

from args import args
from raps.config import ConfigManager, CONFIG_PATH
from raps.schedulers.default import PolicyType
from raps.ui import LayoutManager
from raps.engine import Engine
from raps.flops import FLOPSManager
from raps.power import PowerManager, compute_node_power
from raps.telemetry import Telemetry
from raps.workload import Workload
from raps.utils import convert_to_seconds, next_arrival
from tqdm import tqdm

# Load configurations for each partition
partition_names = args.partitions

print(args.partitions)
if '*' in args.partitions[0]:
    paths = glob.glob(os.path.join(CONFIG_PATH, args.partitions[0]))
    partition_names = [os.path.join(*p.split(os.sep)[-2:]) for p in paths]

configs = [ConfigManager(system_name=partition).get_config() for partition in partition_names]
args_dicts = [{**vars(args), 'config': config} for config in configs]

# Initialize Workload
if args.replay:

    # Currently this assumes that an .npz file has already been created
    # e.g., python main.py --system marconi100 -f ~/data/marconi100/job_table.parquet
    td = Telemetry(**args_dicts[0])
    print(f"Loading {args.replay[0]}...")
    jobs = td.load_snapshot(args.replay[0])
    available_nodes = [config['AVAILABLE_NODES'] for config in configs]
    print("available nodes:", available_nodes)

    # Randomly assign partition
    for job in jobs:
        job['partition'] = random.choices(partition_names, weights=available_nodes, k=1)[0]

    if args.scale:
        for job in tqdm(jobs, desc=f"Scaling jobs to {args.scale} nodes"):
            job['nodes_required'] = random.randint(1, args.scale)
            job['requested_nodes'] = None # Setting to None triggers scheduler to assign nodes

    if args.arrival == 'poisson':
        for job in tqdm(jobs, desc="Rescheduling jobs"):
            partition = job['partition']
            partition_config = configs[partition_names.index(partition)]
            job['requested_nodes'] = None
            job['submit_time'] = next_arrival(1 / partition_config['JOB_ARRIVAL_TIME'])

    elif args.arrival == 'prescribed':
        raise NotImplementedError

else:  # Synthetic workload
    wl = Workload(*configs)

    # Generate jobs based on workload type
    jobs = getattr(wl, args.workload)(num_jobs=args.numjobs)

# Group jobs by partition
jobs_by_partition = {partition: [] for partition in partition_names}
for job in jobs:
    jobs_by_partition[job['partition']].append(job)

# Initialize layout managers for each partition
layout_managers = {}
for i, config in enumerate(configs):
    pm = PowerManager(compute_node_power, **configs[i])
    fm = FLOPSManager(**args_dicts[i])
    sc = Engine(power_manager=pm, flops_manager=fm, cooling_model=None, **args_dicts[i])
    layout_managers[config['system_name']] = LayoutManager(args.layout, engine=sc, debug=args.debug, **config)

# Set simulation timesteps
if args.fastforward:
    fastfoward = convert_to_seconds(args.fastforward)
else:
    fastforward = 0
if args.time:
    timesteps = convert_to_seconds(args.time)
else:
    timesteps = 88200  # Default to 24 hours

timestep_start = fastforward
timestep_end = timestep_start + timesteps

# Create generators for each layout manager
generators = {name: lm.run_stepwise(jobs_by_partition[name], timestep_start=timestep_start, timestep_end=timestep_end)
              for name, lm in layout_managers.items()}

# Step through all generators in lockstep
for timestep in range(timesteps):
    for name, gen in generators.items():
        next(gen)  # Advance each generator

    # Print debug info every UI_UPDATE_FREQ
    if timestep % configs[0]['UI_UPDATE_FREQ'] == 0:  # Assuming same frequency for all partitions
        sys_power = 0
        for name, lm in layout_managers.items():
            sys_util = lm.engine.sys_util_history[-1] if lm.engine.sys_util_history else 0.0
            print(f"[DEBUG] {name} - Timestep {timestep} - Jobs running: {len(lm.engine.running)} - Utilization: {sys_util[1]:.2f}% - Power: {lm.engine.sys_power:.1f}kW")
            sys_power += lm.engine.sys_power
        print(f"system power: {sys_power:.1f}kW")

print("Simulation complete.")

