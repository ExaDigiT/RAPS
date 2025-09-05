from typing import Optional, List
import dataclasses
import pandas as pd
import numpy as np
import threading
import sys
import tty
import termios
import os
import select
import time
import random
from raps.job import Job, JobState
from raps.policy import PolicyType
from raps.utils import (
    summarize_ranges,
    get_current_utilization,
)
from raps.resmgr import ResourceManager
from raps.schedulers import load_scheduler
from raps.power import (
    PowerManager,
    compute_node_power,
    compute_node_power_validate,
    record_power_stats_foreach_job,
    compute_node_power_uncertainties,
    compute_node_power_validate_uncertainties,
)
from raps.network import (
    NetworkModel,
    apply_job_slowdown,
    compute_system_network_stats
)
from raps.telemetry import Telemetry
from raps.cooling import ThermoFluidsModel
from raps.flops import FLOPSManager
from raps.workload import Workload, continuous_job_generation
from raps.account import Accounts
from raps.downtime import Downtime
from raps.weather import Weather
from raps.sim_config import SimConfig
from raps.system_config import SystemConfig
from bisect import bisect_right


@dataclasses.dataclass
class TickData:
    """ Represents the state output from the simulation each tick """
    current_timestep: int
    completed: list[Job]
    killed: list[Job]
    running: list[Job]
    queue: list[Job]
    down_nodes: list[int]
    power_df: Optional[pd.DataFrame]
    p_flops: Optional[float]
    g_flops_w: Optional[float]
    system_util: float
    fmu_inputs: Optional[dict]
    fmu_outputs: Optional[dict]
    num_active_nodes: int
    num_free_nodes: int
    avg_net_tx: float
    avg_net_rx: float
    avg_net_util: float
    slowdown_per_job: float
    node_occupancy: dict[int, int]
    time_delta: int


class SimulationState:
    def __init__(self, time_delta):
        self.paused = False
        self.time_delta = time_delta
        self.lock = threading.Lock()

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused

    def is_paused(self):
        with self.lock:
            return self.paused

    def speed_up(self):
        with self.lock:
            self.time_delta *= 2
            print(f"\n[INFO] time_delta increased to {self.time_delta}", file=sys.stderr)

    def slow_down(self):
        with self.lock:
            if self.time_delta > 1:
                self.time_delta //= 2
                print(f"\n[INFO] time_delta decreased to {self.time_delta}", file=sys.stderr)

    def get_time_delta(self):
        with self.lock:
            return self.time_delta


def keyboard_listener(state):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # or tty.setraw(fd)
        while True:
            # Wait up to 0.1s for input
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                char = os.read(fd, 1).decode()
                if char == 'k' or char == ' ':
                    state.toggle_pause()
                    if state.is_paused():
                        print("\n[PAUSED] Press space or k to resume.", file=sys.stderr)
                    else:
                        print("\n[RESUMED]", file=sys.stderr)
                elif char == 'l' or char == '+':
                    state.speed_up()
                elif char == 'j' or char == '_':
                    state.slow_down()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class Engine:
    """Job scheduling simulation engine."""

    def __init__(self, *,
                 power_manager: PowerManager,
                 flops_manager: FLOPSManager,
                 telemetry: Telemetry,
                 cooling_model: ThermoFluidsModel | None = None,
                 jobs=None,
                 total_initial_jobs=0,
                 # Workload class to generate from for continuous generation
                 continuous_workload: Workload | None = None,
                 accounts=None,
                 sim_config: SimConfig,
                 system_config: SystemConfig,
                 ):
        self.config = system_config.get_legacy()
        self.down_nodes = summarize_ranges(self.config['DOWN_NODES'])
        self.resource_manager = ResourceManager(
            total_nodes=self.config['TOTAL_NODES'],
            down_nodes=self.config['DOWN_NODES'],
            config=self.config
        )
        # Initialize running and queue, etc.
        self.running = []
        self.queue = []
        self.accounts = accounts
        self.telemetry = telemetry
        self.job_history_dict = []
        self.jobs_completed = 0
        self.jobs_killed = 0
        self.total_initial_jobs = total_initial_jobs
        self.current_timestep = 0
        self.cooling_model = cooling_model
        self.sys_power = 0
        self.power_manager = power_manager
        self.flops_manager = flops_manager
        self.debug = sim_config.debug
        self.continuous_workload = continuous_workload
        self.output = sim_config.output
        self.replay = sim_config.replay
        self.downscale = sim_config.downscale  # Factor to downscale the 1s timesteps (power of 10)
        self.simulate_network = sim_config.simulate_network
        self.sys_util_history = []
        self.scheduler_queue_history = []
        self.scheduler_running_history = []
        self.avg_net_tx = []
        self.avg_net_rx = []
        self.net_util_history = []
        self.avg_slowdown_history = []
        self.max_slowdown_history = []
        self.node_occupancy_history = []
        self.downtime = Downtime(first_downtime=sim_config.downtime_first,
                                 downtime_interval=sim_config.downtime_interval,
                                 downtime_length=sim_config.downtime_length,
                                 debug=sim_config.debug,
                                 )

        # Set scheduler type - either based on config or command-line args - defaults to 'default'
        if self.config['multitenant']:
            scheduler_type = 'multitenant'
        else:
            scheduler_type = sim_config.scheduler

        policy_type = sim_config.policy
        backfill_type = sim_config.backfill

        self.scheduler = load_scheduler(scheduler_type)(
            config=self.config,
            policy=policy_type,
            bfpolicy=backfill_type,
            resource_manager=self.resource_manager,
            jobs=jobs
        )
        if sim_config.live:
            assert self.scheduler.policy != PolicyType.REPLAY, \
                "Cannot replay from a live system. Choose a scheduling policy!"
        print(f"Using scheduler: {str(self.scheduler.__class__).split('.')[2]}"
              f", with policy {self.scheduler.policy} "
              f"and backfill {self.scheduler.bfpolicy}")

        if self.simulate_network:
            available_nodes = self.resource_manager.available_nodes
            self.network_model = NetworkModel(
                available_nodes=available_nodes,
                config=self.config,
            )
        else:
            self.network_model = None

    @staticmethod
    def from_sim_config(sim_config: SimConfig, partition: str | None = None):
        if partition:
            system_config_by_name = {s.system_name: s for s in sim_config.system_configs}
            system_config = system_config_by_name.get(partition)
            if not system_config:
                raise ValueError(f"Partition {partition} isn't in SimConfig")
        elif len(sim_config.system_configs) > 1:
            raise ValueError(
                "Engine can only run single-partition simulations. Use MultiPartEngine for " +
                "multi-partition simulations, or pass partition to select the partition to run."
            )
        else:
            system_config = sim_config.system_configs[0]

        # Some temporary backwards/compatibility wrappers
        system_config_dict = system_config.get_legacy()
        sim_config_args = sim_config.get_legacy_args()
        sim_config_dict = sim_config.get_legacy_args_dict()
        sim_config_dict['config'] = system_config_dict
        if partition:
            sim_config_dict["system"] = sim_config.system_name

        if sim_config.seed:
            random.seed(sim_config.seed)
            np.random.seed(sim_config.seed + 1)

        if sim_config.cooling:
            cooling_model = ThermoFluidsModel(**system_config_dict)
            cooling_model.initialize()
            if sim_config.start:
                cooling_model.weather = Weather(sim_config.start, config=system_config_dict)
        else:
            cooling_model = None

        if sim_config.power_scope == 'node':
            if sim_config.uncertainties:
                power_manager = PowerManager(compute_node_power_validate_uncertainties, **system_config_dict)
            else:
                power_manager = PowerManager(compute_node_power_validate, **system_config_dict)
        else:
            if sim_config.uncertainties:
                power_manager = PowerManager(compute_node_power_uncertainties, **system_config_dict)
            else:
                power_manager = PowerManager(compute_node_power, **system_config_dict)

        flops_manager = FLOPSManager(
            config=system_config_dict,
            validate=(sim_config.power_scope == "node"),
        )

        if sim_config.live and not sim_config.replay:
            td = Telemetry(**sim_config_dict)
            workload_data = td.load_from_live_system()
        elif sim_config.replay:
            # TODO: this will have issues if running separate systems or custom systems
            partition_short = partition.split("/")[-1] if partition else None
            td = Telemetry(
                **sim_config_dict,
                partition=partition,
            )
            if partition:
                snap_map = {p.stem: p for p in sim_config.replay[0].glob("*.npz")}
                if len(snap_map) > 0:
                    if partition_short not in snap_map:
                        raise RuntimeError(f"Snapshot '{partition_short}.npz' not in {sim_config.replay[0]}")
                    replay_files = snap_map[partition_short]
                else:
                    replay_files = sim_config.replay
            else:
                replay_files = sim_config.replay

            workload_data = td.load_from_files(replay_files)
        else:  # Synthetic jobs
            wl = Workload(sim_config_args, system_config_dict)
            workload_data = wl.generate_jobs()
            td = Telemetry(**sim_config_dict)

        jobs = workload_data.jobs

        # TODO refactor how stat/end/fastforward/time work
        if sim_config.fastforward is not None:
            workload_data.telemetry_start = workload_data.telemetry_start + sim_config.fastforward

        if sim_config.time is not None:
            workload_data.telemetry_end = workload_data.telemetry_start + sim_config.time

        if sim_config.time_delta is not None:
            time_delta = sim_config.time_delta
        else:
            time_delta = 1

        if sim_config.continuous_job_generation:
            continuous_workload = wl
        else:
            continuous_workload = None

        accounts = None
        if sim_config.accounts:
            job_accounts = Accounts(jobs)
            if sim_config.accounts_json:
                loaded_accounts = Accounts.from_json_filename(sim_config.accounts_json)
                accounts = Accounts.merge(loaded_accounts, job_accounts)
            else:
                accounts = job_accounts

        engine = Engine(
            power_manager=power_manager,
            flops_manager=flops_manager,
            cooling_model=cooling_model,
            continuous_workload=continuous_workload,
            jobs=jobs,
            accounts=accounts,
            telemetry=td,
            sim_config=sim_config,
            system_config=system_config,
        )

        return engine, workload_data, time_delta

    def add_running_jobs_to_queue(self, jobs_to_submit: List):
        """
        Modifies jobs_to_submit and self.queue

        This is a preparatory step and should only be called before the main
        loop of run_simulation.
        Adds running jobs to the queue, and removes them from the jobs_to_submit
        jobs_to_submit still holds the jobs that need be submitted in the future.
        """
        # Build a list of jobs whose start_time is <= current_time.
        eligible_jobs = [job for job in jobs_to_submit if
                         job.start_time is not None
                         and job.start_time < self.current_timestep]
        # Remove those jobs from jobs_to_submit:
        jobs_to_submit[:] = [job for job in jobs_to_submit if
                             job.start_time is None
                             or job.start_time >= self.current_timestep]
        # Convert them to Job instances and build list of eligible jobs.
        self.queue += eligible_jobs

    def add_eligible_jobs_to_queue(self, jobs_to_submit: List):
        """
        Modifies jobs_to_submit and self.queue

        Adds eligible jobs to the queue, and removes them from the jobs_to_submit
        jobs_to_submit still holds the jobs that need be submitted in the future.
        returns
        - true if new jobs are present
        - false if no new jobs are present
        """
        # Build a list of jobs whose submit_time is <= current_time.
        eligible_jobs = [job for job in jobs_to_submit if job.submit_time <= self.current_timestep]
        # Remove those jobs from jobs_to_submit:
        jobs_to_submit[:] = [job for job in jobs_to_submit if job.submit_time > self.current_timestep]
        # Convert them to Job instances and build list of eligible jobs.
        self.queue += eligible_jobs
        if eligible_jobs != []:
            return True
        else:
            return False

    def prepare_timestep(self, *, replay: bool = True, jobs):
        # 1 identify completed jobs
        # 2 Check continuous job generation
        # 3 Simulate node failure # Defunct feature!
        # 4 Simulate downtime
        # 5 Update active and free nodes

        # 1 Identify Completed Jobs
        completed_jobs = [job for job in self.running if
                          job.end_time is not None and job.end_time <= self.current_timestep]

        # Update Completed Jobs, their account and  and Free resources.
        for job in completed_jobs:
            self.power_manager.set_idle(job.scheduled_nodes)
            job.current_state = JobState.COMPLETED
            job.end_time = self.current_timestep
            self.running.remove(job)
            self.jobs_completed += 1
            job_stats = job.statistics()
            if self.accounts:
                self.accounts.update_account_statistics(job_stats)
            self.job_history_dict.append(job_stats.__dict__)
            # Free the nodes via the resource manager.
            self.resource_manager.free_nodes_from_job(job)

        killed_jobs = [job for job in self.running if
                       job.end_time is not None and job.start_time + job.time_limit <= self.current_timestep]

        for job in killed_jobs:
            self.power_manager.set_idle(job.scheduled_nodes)
            job.current_state = JobState.TIMEOUT
            job.end_time = self.current_timestep

            self.running.remove(job)
            self.jobs_killed += 1
            job_stats = job.statistics()
            if self.accounts:
                self.accounts.update_account_statistics(job_stats)
            self.job_history_dict.append(job_stats.__dict__)
            # Free the nodes via the resource manager.
            self.resource_manager.free_nodes_from_job(job)

        # 2 Check continuous job generation
        if self.continuous_workload is not None:  # Experimental
            continuous_job_generation(engine=self, timestep=self.current_timestep, jobs=jobs)

        # 3 Simulate node failure
        if not replay:
            newly_downed_nodes = self.resource_manager.node_failure(self.config['MTBF'])
            for node in newly_downed_nodes:
                self.power_manager.set_idle(node)
        else:
            newly_downed_nodes = []

        need_reschedule = False
        # 4 Simulate downtime
        need_reschedule = self.downtime.check_and_trigger(timestep=self.current_timestep, engine=self)

        # 5 Update active/free nodes based on core/GPU utilization
        if self.config['multitenant']:
            # #total_cpu_cores = sum(node['total_cpu_cores'] for node in self.resource_manager.nodes)
            # #total_gpu_units = sum(node['total_gpu_units'] for node in self.resource_manager.nodes)
            # #available_cpu_cores = sum(node['available_cpu_cores'] for node in self.resource_manager.nodes)
            # #available_gpu_units = sum(node['available_gpu_units'] for node in self.resource_manager.nodes)

            self.num_free_nodes = len([node for node in self.resource_manager.nodes if
                                       not node['is_down']
                                       and node['available_cpu_cores'] == node['total_cpu_cores']
                                       and node['available_gpu_units'] == node['total_gpu_units']])
            self.num_active_nodes = len([node for node in self.resource_manager.nodes if
                                         not node['is_down']
                                         and (node['available_cpu_cores'] < node['total_cpu_cores']
                                              or node['available_gpu_units'] < node['total_gpu_units'])])

            # Update system utilization history
            self.resource_manager.update_system_utilization(self.current_timestep, self.running)
        else:
            # Whole-node allocator
            self.num_free_nodes = len(self.resource_manager.available_nodes)
            self.num_active_nodes = self.config['TOTAL_NODES'] \
                - len(self.resource_manager.available_nodes) \
                - len(self.resource_manager.down_nodes)
        self.down_nodes = self.resource_manager.down_nodes
        # TODO This should only be managed in the resource manager!

        return completed_jobs, killed_jobs, newly_downed_nodes, need_reschedule

    def complete_timestep(self, *,
                          actively_considered_jobs: List,
                          all_jobs: List,
                          replay: bool,
                          autoshutdown: bool,
                          cursor: int):
        # 1 update running time of all running jobs
        # 2 update the current_timestep of the engine (this serves as reference for most computations)
        # 3 Check if simulation should shutdown

        # update Running time
        for job in self.running:
            if job.current_state == JobState.RUNNING:
                job.running_time = self.current_timestep - job.start_time

        # Stop the simulation if no more jobs are running or in the queue or in the job list.
        if autoshutdown and \
           len(self.queue) == 0 and \
           len(self.running) == 0 and \
           not replay and \
           len(all_jobs) == cursor and \
           len(actively_considered_jobs) == 0:
            if self.debug:
                print(f"Simulaiton completed early: {self.config['system_name']} - "
                      f"Stopping simulation at time {self.current_timestep}. "
                      f"Simulation ran for {self.current_timestep - self.timestep_start}")
            simulation_complete = True
        else:
            simulation_complete = False
            self.current_timestep += 1  # Update the current time every timestep

        return simulation_complete

    def tick(self, *, time_delta=1, replay=False):
        # Tick runs all simulations of interest at the given time delta interval.
        #
        # The simulations which are needed for simulations consistency at each time step
        # (inside: the main simulation loop of run_simulation) are not part of tick.
        #
        # Tick contains:
        # For each running job:
        #  - CPU utilization
        #  - GPU utilization
        #  - Network utilization
        #
        # From these the systems (across all nodes)
        #  - System Utilization
        #  - Power
        #  - Cooling
        #  - System Performance
        # is simulated.

        scheduled_nodes = []
        cpu_utils = []
        gpu_utils = []
        net_congs = []
        net_utils = []
        net_tx_list = []
        net_rx_list = []

        slowdown_factors = []

        for job in self.running:

            job.running_time = self.current_timestep - job.start_time

            if job.current_state != JobState.RUNNING:
                raise ValueError(
                    f"Job {job.id} is in running list, " +
                    "but state is not RUNNING: job.state == {job.current_state}"
                )
            else:  # if job.state == JobState.RUNNING:
                # Error checks
                if job.running_time > job.time_limit and job.end_time is not None:
                    raise Exception(f"Job exceded time limit! "
                                    f"{job.running_time} > {job.time_limit}"
                                    f"\n{job}"
                                    f"\nCurrent timestep:{self.current_timestep-self.timestep_start} (rel)"
                                    )
                if replay and job.running_time > job.expected_run_time:
                    raise Exception(f"Job should have ended in replay! "
                                    f" {job.running_time} > {job.expected_run_time}"
                                    f"\n{job}"
                                    f"\nCurrent timestep:{self.current_timestep-self.timestep_start} (rel)"
                                    )

                # Aggregate scheduled nodes
                scheduled_nodes.append(job.scheduled_nodes)

                # Get CPU utilization
                cpu_util = get_current_utilization(job.cpu_trace, job)
                cpu_utils.append(cpu_util)
                # Percentage Utilization!

                # Get GPU utilization
                gpu_util = get_current_utilization(job.gpu_trace, job)
                gpu_utils.append(gpu_util)
                # Percentage Utilization!

                # Simulate network utilization
                if self.simulate_network:

                    net_util, net_cong, net_tx, net_rx, max_throughput = \
                        self.network_model.simulate_network_utilization(job=job, debug=self.debug)

                    net_utils.append(net_util)
                    net_congs.append(net_cong)
                    net_tx_list.append(net_tx)
                    net_rx_list.append(net_rx)

                else:
                    net_util, net_cong, net_tx, net_rx = 0.0, 0.0, 0.0, 0.0
                    max_throughput = 0
                    net_utils.append(net_util)
                    net_congs.append(net_cong)
                    net_tx_list.append(net_tx)
                    net_rx_list.append(net_rx)

                # Apply slowdowns
                slowdown_factor = apply_job_slowdown(job=job,
                                                     max_throughput=max_throughput,
                                                     net_util=net_util,
                                                     net_cong=net_cong,
                                                     net_tx=net_tx,
                                                     net_rx=net_rx,
                                                     debug=self.debug)
                slowdown_factors.append(slowdown_factor)

        # All required values for each jobs have been an collected.
        # Continue with calculations for the whole system:

        # System Utilization Statistics
        system_util = self.num_active_nodes / self.config['AVAILABLE_NODES'] * 100
        self.record_util_stats(system_util=system_util)

        # System Power
        if self.power_manager:  # Power is always simulated
            power_df, rack_power, total_power_kw, total_loss_kw, jobs_power = \
                self.power_manager.simulate_power(running_jobs=self.running,
                                                  scheduled_nodes=scheduled_nodes,
                                                  cpu_utils=cpu_utils,
                                                  gpu_utils=gpu_utils,
                                                  net_utils=net_utils)

            # Unclear what jobs_power is!
            self.record_power_stats(time_delta=time_delta,
                                    total_power_kw=total_power_kw,
                                    total_loss_kw=total_loss_kw,
                                    jobs_power=jobs_power)
        else:
            power_df = None

        # System Cooling
        if self.cooling_model:
            cooling_inputs, cooling_outputs = self.cooling_model.simulate_cooling(rack_power=rack_power,
                                                                                  engine=self)
        else:
            cooling_inputs, cooling_outputs = None, None

        # System total Flops
        if self.flops_manager:
            pflops, gflops_per_watt = self.flops_manager.simulate_flops(scheduled_nodes=scheduled_nodes,
                                                                        cpu_util=cpu_utils,
                                                                        gpu_util=gpu_utils,
                                                                        total_power_kw=total_power_kw)

        # System Network
        if self.network_model:
            avg_tx, avg_rx, avg_net = compute_system_network_stats(net_utils=net_utils,
                                                                   net_tx_list=net_tx_list,
                                                                   net_rx_list=net_rx_list,
                                                                   slowdown_factors=slowdown_factors
                                                                   )
            slowdown_per_job = sum(slowdown_factors)/len(slowdown_factors) if len(slowdown_factors) != 0 else 0
            self.record_network_stats(avg_tx=avg_tx,
                                      avg_rx=avg_rx,
                                      avg_net=avg_net)
        else:
            avg_tx, avg_rx, avg_net = None, None, None
            slowdown_per_job = 0

        # Continue with System Simulation

        # Calculate node occupancy
        node_occupancy = {node['id']: 0 for node in self.resource_manager.nodes}  # Initialize even if no running jobs
        for job in self.running:
            if job.scheduled_nodes:
                node_id = job.scheduled_nodes[0]  # Assuming one node per job for multitenancy
                node_occupancy[node_id] += 1

        self.node_occupancy_history.append(node_occupancy)

        tick_data = TickData(
            current_timestep=self.current_timestep,
            completed=None,
            killed=None,
            running=self.running,
            queue=self.queue,
            down_nodes=self.down_nodes,
            power_df=power_df,
            p_flops=pflops,
            g_flops_w=gflops_per_watt,
            system_util=system_util,
            fmu_inputs=cooling_inputs,
            fmu_outputs=cooling_outputs,
            num_active_nodes=self.num_active_nodes,
            num_free_nodes=self.num_free_nodes,
            avg_net_tx=avg_tx,
            avg_net_rx=avg_rx,
            avg_net_util=avg_net,
            slowdown_per_job=slowdown_per_job,
            node_occupancy=node_occupancy,
            time_delta=time_delta
        )
        return tick_data

    def prepare_system_state(self, *, all_jobs: List, timestep_start, timestep_end):
        # Set engine timesteps
        self.timestep_start = timestep_start
        self.current_timestep = timestep_start
        self.timestep_end = timestep_end

        # Modifies Jobs object
        # Keep only jobs that have not yet ended and that have a chance to start
        all_jobs[:] = [job for job in all_jobs if
                       job.submit_time < timestep_end
                       and ((job.end_time is not None
                             and job.end_time >= timestep_start)
                            or job.end_time is None)
                       ]
        all_jobs.sort(key=lambda j: j.submit_time)

        self.add_running_jobs_to_queue(all_jobs)
        # Set policy to replay and no backfill to get the original prefilled placement.
        target_policy = self.scheduler.policy
        self.scheduler.policy = PolicyType.REPLAY
        target_bfpolicy = self.scheduler.bfpolicy
        self.scheduler.bfpolicy = None

        # Now process job queue one by one (needed to get the start_time right!)
        for job in self.queue[:]:  # operate over a slice copy to be able to remove jobs from queue if placed.
            self.scheduler.schedule([job], self.running, job.start_time, accounts=self.accounts, sorted=True)
            self.queue.remove(job)
        if self.replay and len(self.queue) != 0:
            raise ValueError(
                f"Something went wrong! Not all jobs could be placed!\nPotential confligt in queue:\n{self.queue}")
        # Restore the target policy and backfill for the remainder of the simulation.
        self.scheduler.policy = target_policy
        self.scheduler.bfpolicy = target_bfpolicy

    def run_simulation(self, jobs, timestep_start, timestep_end, time_delta=1, autoshutdown=False):
        """Generator that yields after each simulation tick."""

        if self.scheduler.policy == PolicyType.REPLAY:
            replay = True
        else:
            replay = False

        if self.debug:
            print(f"[DEBUG] run_simulation: Initial jobs count: {len(jobs)}")
            if jobs:
                print("[DEBUG] run_simulation: First job submit_time: "
                      f"{jobs[0].submit_time}, start_time: {jobs[0].start_time}")

        # Set times and place jobs that are currently running, onto the system.
        self.prepare_system_state(all_jobs=jobs, timestep_start=timestep_start, timestep_end=timestep_end)

        # Process jobs in batches for better performance of timestep loop
        all_jobs = jobs.copy()
        submit_times = [j.submit_time for j in all_jobs]
        cursor = 0

        jobs = []
        # Batch Jobs into 6h windows based on submit_time or twice the time_delta if larger
        batch_window = max(60 * 60 * 6, 2 * time_delta)  # at least 6h

        sim_state = SimulationState(time_delta)
        # listener_thread = threading.Thread(target=keyboard_listener, args=(sim_state,), daemon=True)
        # listener_thread.start()

        while self.current_timestep < self.timestep_end:  # Runs every seconds!

            if sim_state.is_paused():
                time.sleep(0.1)
                continue

            current_time_delta = sim_state.get_time_delta()

            if (self.current_timestep % batch_window == 0) or (self.current_timestep == timestep_start):
                # Add jobs that are within the batching window and remove them from all jobs
                # jobs += [job for job in all_jobs if job.submit_time <= self.current_timestep + batch_window]
                # all_jobs[:] = [job for job in all_jobs if job.submit_time > self.current_timestep + batch_window]
                cutoff = self.current_timestep + batch_window
                r = bisect_right(submit_times, cutoff, lo=cursor)
                if r > cursor:
                    jobs.extend(all_jobs[cursor:r])
                    cursor = r

            # 1. Prepare Timestep:
            completed_jobs, killed_jobs, newly_downed_nodes, need_reschedule = \
                self.prepare_timestep(jobs=jobs)

            # 2. Identify eligible jobs and add them to the queue.
            has_new_additions = self.add_eligible_jobs_to_queue(jobs)

            # 3. Schedule jobs that are now in the queue.
            if completed_jobs != [] \
                or killed_jobs != [] \
                or newly_downed_nodes != [] \
                or has_new_additions \
                    or need_reschedule:
                self.scheduler.schedule(self.queue, self.running,
                                        self.current_timestep,
                                        accounts=self.accounts,
                                        sorted=(not has_new_additions))

            if self.debug and self.current_timestep % self.config['UI_UPDATE_FREQ'] == 0:
                print(".", end="", flush=True)

            # 4. Run tick only at specified time_delta
            if 0 == (self.current_timestep % current_time_delta) \
               and ((current_time_delta == 1
                    and self.current_timestep % self.config['POWER_UPDATE_FREQ'] == 0)
                    or (current_time_delta != 1 or self.downscale != 1)
                    ):
                tick_data = self.tick(time_delta=current_time_delta, replay=replay)
                tick_data.completed = completed_jobs
                tick_data.killed = completed_jobs
            else:
                tick_data = None

            # 5. Complete the timestep
            simulation_done = self.complete_timestep(actively_considered_jobs=jobs,
                                                     all_jobs=all_jobs,
                                                     replay=replay,
                                                     autoshutdown=autoshutdown,
                                                     cursor=cursor)
            if simulation_done:
                break
            yield tick_data

    def get_job_history_dict(self):
        return self.job_history_dict

    def get_scheduler_queue_history(self):
        return self.scheduler_queue_history

    def get_scheduler_running_history(self):
        return self.scheduler_running_history

    def record_util_stats(self, *, system_util):
        self.sys_util_history.append((self.current_timestep, system_util))
        self.scheduler_queue_history.append(len(self.running))
        self.scheduler_running_history.append(len(self.queue))

    def record_network_stats(self, *,
                             avg_tx,
                             avg_rx,
                             avg_net
                             ):
        self.avg_net_tx.append(avg_tx)
        self.avg_net_rx.append(avg_rx)
        self.net_util_history.append(avg_net)

    def record_power_stats(self, *, time_delta, total_power_kw, total_loss_kw, jobs_power):
        if (time_delta == 1 and self.current_timestep % self.config['POWER_UPDATE_FREQ'] == 0) or time_delta != 1:
            # First job specific
            record_power_stats_foreach_job(running_jobs=self.running, jobs_power=jobs_power)
            # power manager
            self.power_manager.history.append((self.current_timestep, total_power_kw))
            self.power_manager.loss_history.append((self.current_timestep, total_loss_kw))
        # engine
        self.sys_power = total_power_kw
