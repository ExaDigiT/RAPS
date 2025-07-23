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
from raps.utils import create_casename, convert_to_seconds, next_arrival
from raps.stats import get_engine_stats, get_job_stats, get_scheduler_stats, get_network_stats
from tqdm import tqdm

# Load configurations for each partition
partition_names = args.partitions

print(args.partitions)
if '*' in args.partitions[0]:
    paths = glob.glob(os.path.join(CONFIG_PATH, args.partitions[0]))
    partition_names = [os.path.join(*p.split(os.sep)[-2:]) for p in paths]

configs = [ConfigManager(system_name=partition).get_config() for partition in partition_names]
args_dicts = [
       {**vars(args), 'config': config, 'partition': partition_names[i]}
       for i, config in enumerate(configs)
   ]

# Initialize Workload
if args.replay:

    jobs_by_partition = {}
    t0_by_partition = {}
    t1_by_partition = {}

    if args.replay[0].endswith('.npz'):
        # snapshot mode: pick the right .npz for each partition
        snap_map = { os.path.basename(p): p for p in args.replay }
        for ad in args_dicts:
            part = ad['partition']                        # e.g. 'mit_supercloud/part-cpu'
            short = part.split('/')[-1]                   # 'part-cpu'
            snap_file = f"{short}.npz"
            if snap_file not in snap_map:
                raise RuntimeError(f"Snapshot '{snap_file}' not in {args.replay}")
            td = Telemetry(**ad)
            print(f"[{part}] loading snapshot {snap_file} …")
            jobs_part, t0, t1, args_from_file = td.load_snapshot(snap_map[snap_file])
            jobs_by_partition[part] = jobs_part
    else:
        # raw load_data mode
        for ad in args_dicts:
            part = ad['partition']
            td = Telemetry(**ad)
            print(f"[{part}] loading traces from {args.replay[0]} …")
            jobs_part, t0, t1, args_from_file = td.load_data(args.replay)
            jobs_by_partition[part] = jobs_part
            td.save_snapshot(jobs_part, t0, t1, args_from_file, filename=part.split('/')[-1])

    # --- report how many jobs per partition ---
    for part, jl in jobs_by_partition.items():
        print(f"[INFO] Partition '{part}': {len(jl)} jobs loaded")

    # now flatten into a single job list (or keep separate for your engine)
    all_jobs_flat = []
    for part in partition_names:
        for job in jobs_by_partition[part]:
            job['partition'] = part
            all_jobs_flat.append(job)

    total_initial_jobs = len(all_jobs_flat)
    jobs = all_jobs_flat

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

else:  # Synthetic workload
    wl = Workload(*configs)

    total_initial_jobs = args.numjobs

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
    sc = Engine(power_manager=pm, flops_manager=fm, cooling_model=None, jobs=jobs_by_partition[config['system_name']], total_initial_jobs=total_initial_jobs, **args_dicts[i])
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
            sys_util = lm.engine.sys_util_history[-1] if lm.engine.sys_util_history else (0, 0.0)
            allocated_cores = lm.engine.resource_manager.allocated_cpu_cores
            print(f"[DEBUG] {name} - Timestep {timestep} - Jobs running: {len(lm.engine.running)} - Utilization: {sys_util[1]:.2f}% - Allocated Cores: {allocated_cores} - Power: {lm.engine.sys_power:.1f}kW")
            sys_power += lm.engine.sys_power
        print(f"system power: {sys_power:.1f}kW")

print("Simulation complete.")

# Print statistics for each partition
for name, lm in layout_managers.items():
    print(f"\n--- Simulation Report for Partition: {name} ---")
    simulation_stats = lm.engine.get_stats()
    for key, value in simulation_stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------------------------------")
