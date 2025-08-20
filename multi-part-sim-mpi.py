"""
MPI-enabled driver for simulating multi-partition RAPS systems.
Distributes partitions across ranks with mpi4py for parallel run.
Supports telemetry replay or synthetic workloads with per-rank
power, FLOPS, and scheduling models. Outputs debug and summary
stats for heterogeneous systems (e.g., LUMI, Setonix, Adastra).
"""

from tqdm import tqdm
from mpi4py import MPI
from raps.utils import convert_to_seconds, next_arrival
from raps.workload import Workload
from raps.telemetry import Telemetry
from raps.power import PowerManager, compute_node_power
from raps.flops import FLOPSManager
from raps.engine import Engine
from raps.ui import LayoutManager
from raps.config import get_system_config, CONFIG_PATH
from args import args
import random
import os
import glob
from raps.helpers import check_python_version
check_python_version()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Expand “partitions” (on rank 0) if the user used a glob:
    if rank == 0:
        partition_names = args.partitions
        if '*' in partition_names[0]:
            paths = glob.glob(os.path.join(CONFIG_PATH, partition_names[0]))
            partition_names = [os.path.join(*p.split(os.sep)[-2:]) for p in paths]
    else:
        partition_names = None

    # 2) Broadcast the final list of partition_names to everyone
    partition_names = comm.bcast(partition_names, root=0)

    # 3) Load configs for every partition (all ranks do this)
    configs = [get_system_config(p).get_legacy() for p in partition_names]
    args_dicts = [{**vars(args), 'config': cfg} for cfg in configs]

    # 4) Each rank decides which partition‐indices it owns (round-robin):
    local_partition_indices = [i for i in range(len(partition_names)) if (i % size) == rank]
    local_partition_names = [partition_names[i] for i in local_partition_indices]
    # local_configs = [configs[i] for i in local_partition_indices]   # Unused
    # local_args_dicts = [args_dicts[i] for i in local_partition_indices]  # Unused

    # 5) Rank 0 builds (or loads) the entire job list, assigns partitions, groups by partition,
    #    then scatters exactly those jobs to each rank. Other ranks just sit in the scatter:
    if rank == 0:
        # --- a) “REPLAY” branch?
        if args.replay:
            td = Telemetry(**args_dicts[0])
            print(f"[rank 0] Loading telemetry from {args.replay[0]}…")
            jobs_full = td.load_snapshot(args.replay[0])
            available_nodes = [c['AVAILABLE_NODES'] for c in configs]
            for job in jobs_full:
                job['partition'] = random.choices(partition_names, weights=available_nodes, k=1)[0]
            if args.scale:
                for job in tqdm(jobs_full, desc="[rank 0] Scaling jobs…"):
                    job['nodes_required'] = random.randint(1, args.scale)
                    job['requested_nodes'] = None
            if args.arrival == 'poisson':
                for job in tqdm(jobs_full, desc="[rank 0] Rescheduling arrivals…"):
                    p_name = job['partition']
                    p_cfg = configs[partition_names.index(p_name)]
                    job['requested_nodes'] = None
                    job['submit_time'] = next_arrival(1 / p_cfg['JOB_ARRIVAL_TIME'])

        # --- b) “SYNTHETIC WORKLOAD” branch:
        else:
            wl = Workload(*configs)
            jobs_full = getattr(wl, args.workload)(num_jobs=args.numjobs)
            available_nodes = [c['AVAILABLE_NODES'] for c in configs]
            for job in jobs_full:
                job['partition'] = random.choices(partition_names, weights=available_nodes, k=1)[0]

        # --- c) Group “jobs_full” by partition name:
        jobs_by_partition = {p: [] for p in partition_names}
        for job in jobs_full:
            jobs_by_partition[job['partition']].append(job)

        # --- d) Build a list-of-lists, one list per rank, containing the union
        #     of all jobs for that rank’s partitions:
        jobs_for_rank = [[] for _ in range(size)]
        for p_idx, p_name in enumerate(partition_names):
            tgt = p_idx % size
            jobs_for_rank[tgt].extend(jobs_by_partition[p_name])
    else:
        jobs_for_rank = None

    # 6) Scatter the per-rank job lists:
    local_jobs = comm.scatter(jobs_for_rank, root=0)

    # 7) Re‐group each rank’s “local_jobs” into a dict keyed by its local_partition_names:
    local_jobs_by_partition = {p: [] for p in local_partition_names}
    for job in local_jobs:
        local_jobs_by_partition[job['partition']].append(job)

    # 8) Build one LayoutManager (and Engine/PowerManager/FLOPSManager) per local partition:
    layout_managers = {}
    for idx, p_name in enumerate(local_partition_names):
        global_idx = local_partition_indices[idx]
        cfg = configs[global_idx]
        ad = args_dicts[global_idx]

        pm = PowerManager(compute_node_power, **cfg)
        fm = FLOPSManager(**ad)
        sc = Engine(power_manager=pm, flops_manager=fm,
                    cooling_model=None, **ad)

        layout_managers[p_name] = LayoutManager(args.layout,
                                                engine=sc,
                                                debug=args.debug,
                                                **cfg)

    # 9) Compute timestep_start / timestep_end (all ranks agree):
    if args.fastforward:
        fastforward = convert_to_seconds(args.fastforward)
    else:
        fastforward = 0

    if args.time:
        timesteps = convert_to_seconds(args.time)
    else:
        timesteps = 88200   # default 24 hours

    timestep_start = fastforward
    timestep_end = timestep_start + timesteps

    # 10) Build a generator for each partition that this rank owns:
    local_generators = {}
    for p_name in local_partition_names:
        gen = layout_managers[p_name].run_stepwise(
            local_jobs_by_partition[p_name],
            timestep_start=timestep_start,
            timestep_end=timestep_end
        )
        local_generators[p_name] = gen

    # 11) Main simulation loop (every rank steps its own partitions in lockstep):
    UIF = configs[0]['UI_UPDATE_FREQ']  # assume same for all configs
    for t in range(timesteps):
        # --- a) Advance each local partition’s generator
        for gen in local_generators.values():
            try:
                next(gen)
            except StopIteration:
                pass

        # --- b) Every UI_UPDATE_FREQ, do per-rank prints + one global reduction
        if (t % UIF) == 0:
            # 1) sum our local sys_power
            local_sys_power = sum(lm.engine.sys_power for lm in layout_managers.values())

            # 2) print *our* partition‐level info now (so rank 0 and rank 1 will both print):
            for p_name, lm in layout_managers.items():
                sys_util = lm.engine.sys_util_history[-1] if lm.engine.sys_util_history else 0.0
                print(f"[DEBUG][rank {rank}] {p_name} – Timestep {t} – "
                      f"Jobs running: {len(lm.engine.running)} – "
                      f"Utilization: {sys_util[1]:.2f}% – "
                      f"Power: {lm.engine.sys_power:.1f}kW")

            # 3) do an MPI reduce so that rank 0 knows the total across all ranks:
            total_sys_power = comm.reduce(local_sys_power, op=MPI.SUM, root=0)
            if rank == 0:
                print(f"[DEBUG][rank {rank}] TOTAL system power (all partitions): {total_sys_power:.1f}kW")

    # 12) Final barrier + exit message on rank 0
    comm.Barrier()
    if rank == 0:
        print("Simulation complete (all ranks).")


if __name__ == "__main__":
    main()
