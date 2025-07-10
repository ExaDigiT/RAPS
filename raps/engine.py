from typing import Optional, List
import dataclasses
import pandas as pd

from raps.job import Job, JobState
from raps.policy import PolicyType
from raps.utils import summarize_ranges, expand_ranges, get_current_utilization
from raps.resmgr import ResourceManager
from raps.schedulers import load_scheduler
from raps.power import record_power_stats_foreach_job
from raps.network import (
    NetworkModel,
    apply_job_slowdown,
    compute_system_network_stats
)


@dataclasses.dataclass
class TickData:
    """ Represents the state output from the simulation each tick """
    current_time: int
    completed: list[Job]
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


class Engine:
    """Job scheduling simulation engine."""

    def __init__(self, *, power_manager, flops_manager, cooling_model=None, config, jobs=None, **kwargs):
        self.config = config
        self.down_nodes = summarize_ranges(self.config['DOWN_NODES'])
        self.resource_manager = ResourceManager(
            total_nodes=self.config['TOTAL_NODES'],
            down_nodes=self.config['DOWN_NODES']
        )
        # Initialize running and queue, etc.
        self.running = []
        self.queue = []
        self.accounts = None
        self.job_history_dict = []
        self.jobs_completed = 0
        self.current_time = 0
        self.cooling_model = cooling_model
        self.sys_power = 0
        self.power_manager = power_manager
        self.flops_manager = flops_manager
        self.debug = kwargs.get('debug')
        self.output = kwargs.get('output')
        self.replay = kwargs.get('replay')
        self.simulate_network = kwargs.get('simulate_network')
        self.sys_util_history = []
        self.scheduler_queue_history = []
        self.scheduler_running_history = []
        self.avg_net_tx = []
        self.avg_net_rx = []
        self.net_util_history = []
        self.avg_slowdown_history = []
        self.max_slowdown_history = []

        # Get scheduler type from command-line args or default
        scheduler_type = kwargs.get('scheduler', 'default')
        policy_type = kwargs.get('policy')
        backfill_type = kwargs.get('backfill')

        self.scheduler = load_scheduler(scheduler_type)(
            config=self.config,
            policy=policy_type,
            bfpolicy=backfill_type,
            resource_manager=self.resource_manager,
            jobs=jobs
        )
        print(f"Using scheduler: {str(self.scheduler.__class__).split('.')[2]}"\
              f", with policy {self.scheduler.policy} "\
              f"and backfill {self.scheduler.bfpolicy}")

        if self.simulate_network:
            available_nodes = self.resource_manager.available_nodes
            self.network_model = NetworkModel(available_nodes=available_nodes,config=config,kwargs=kwargs)
        else:
            self.network_model = None

    def add_running_jobs_to_queue(self, jobs_to_submit: List):
        """
        Modifies jobs_to_submit and self.queue

        This is a preparatory step and should only be called before the main
        loop of run_simulation.
        Adds running jobs to the queue, and removes them from the jobs_to_submit
        jobs_to_submit still holds the jobs that need be submitted in the future.
        """
        # Build a list of jobs whose start_time is <= current_time.
        eligible_jobs = [job for job in jobs_to_submit if job.start_time < self.current_time]
        # Remove those jobs from jobs_to_submit:
        jobs_to_submit[:] = [job for job in jobs_to_submit if job.start_time >= self.current_time]
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
        eligible_jobs = [job for job in jobs_to_submit if job.submit_time <= self.current_time]
        # Remove those jobs from jobs_to_submit:
        jobs_to_submit[:] = [job for job in jobs_to_submit if job.submit_time > self.current_time]
        # Convert them to Job instances and build list of eligible jobs.
        self.queue += eligible_jobs
        if eligible_jobs != []:
            return True
        else:
            return False

    def prepare_timestep(self, replay:bool = True):
        # 1 identify completed jobs
        # 2 Simulate node failure # Defunct feature!
        # 3 Update active and free nodes

        # Identify Completed Jobs
        completed_jobs = [job for job in self.running if job.end_time is not None and job.end_time <= self.current_time]
        # Update Completed Jobs, their account and  and Free resources.
        for job in completed_jobs:
            self.power_manager.set_idle(job.scheduled_nodes)
            job.state = JobState.COMPLETED

            self.running.remove(job)
            self.jobs_completed += 1
            job_stats = job.statistics()
            if self.accounts:
                self.accounts.update_account_statistics(job_stats)
            self.job_history_dict.append(job_stats.__dict__)
            # Free the nodes via the resource manager.
            self.resource_manager.free_nodes_from_job(job)

        if not replay:
            # Simulate node failure
            newly_downed_nodes = self.resource_manager.node_failure(self.config['MTBF'])
            for node in newly_downed_nodes:
                self.power_manager.set_idle(node)
        else:
            newly_downed_nodes = []

        # Update active/free nodes
        self.num_free_nodes = len(self.resource_manager.available_nodes)
        self.num_active_nodes = self.config['TOTAL_NODES'] \
                              - len(self.resource_manager.available_nodes) \
                              - len(self.resource_manager.down_nodes)

        return completed_jobs, newly_downed_nodes

    def complete_timestep(self, autoshutdown, all_jobs:List, jobs:List):
        # 1 update running time of all running jobs
        # 2 update the current_time of the engine (this serves as reference for most computations)
        # 3 Check if simulation should shutdown

        #update Running time
        for job in self.running:
            if job.state == JobState.RUNNING:
                job.running_time = self.current_time - job.start_time

        self.current_time += 1  # Update the current time every timestep

        # Stop the simulation if no more jobs are running or in the queue or in the job list.
        if autoshutdown and \
           len(self.queue) == 0 and \
           len(self.running) == 0 and \
           not self.replay and \
           len(all_jobs) == 0 and \
           len(jobs) == 0:
            print(f"[DEBUG] {self.config['system_name']} - Stopping simulation at time {self.current_time}")
            simulation_complete = True
        else:
            simulation_complete = False
        return simulation_complete

    def tick(self, *, time_delta=1):
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
        if self.debug:
            print(f"Current Time: {self.current_time}")

        slowdown_factors = []

        for job in self.running:
            if job.end_time == self.current_time:
                job.state = JobState.COMPLETED

        for job in self.running:
            if self.debug:
                print(f"JobID: {job.id}")

            if job.state != JobState.RUNNING:
                raise ValueError(f"Job is in running list, but state is not RUNNING: job.state == {job.state}")
            else:  # if job.state == JobState.RUNNING:
                # Error checks
                if job.running_time > job.wall_time:
                    raise Exception(f"Job should have ended already!\n\
                                       {job.running_time} > {job.wall_time}\
                                    ")
                # Aggregate scheduled nodes
                scheduled_nodes.append(job.scheduled_nodes)

                # Get CPU utilization
                cpu_util = get_current_utilization(job.cpu_trace, job)
                cpu_utils.append(cpu_util)

                # Get GPU utilizaiton
                gpu_util = get_current_utilization(job.gpu_trace, job)
                gpu_utils.append(gpu_util)

                # Simulate network utilization
                if self.simulate_network:

                    net_util, net_cong, net_tx, net_rx, max_throughput = self.network_model.simulate_network_utilization(job=job,debug=self.debug)

                    net_utils.append(net_util)
                    net_congs.append(net_cong)
                    net_tx_list.append(net_tx)
                    net_rx_list.append(net_rx)

                else:
                    net_util, net_cong, net_tx, net_rx = 0.0,0.0,0.0,0.0
                    max_throughput = 0
                    net_utils.append(net_util)
                    net_congs.append(net_cong)
                    net_tx_list.append(net_tx)
                    net_rx_list.append(net_rx)

                #Apply slowdowns
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
            cooling_inputs, cooling_outputs = self.cooling_model.simulate_cooling(self.cooling_model, rack_power)
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
        else:
            avg_tx, avg_rx, avg_net = None,None,None
        self.record_network_stats(avg_tx=avg_tx,
                                  avg_rx=avg_rx,
                                  avg_net=avg_net)

        # Continue with System Simulation
        tick_data = TickData(
            current_time=self.current_time,
            completed=None,
            running=self.running,
            queue=self.queue,
            down_nodes=expand_ranges(self.down_nodes[1:]),
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
            slowdown_per_job=0
        )
        return tick_data

    def prepare_system_state(self, all_jobs:List, timestep_start, timestep_end, replay:bool):
        # Modifies Jobs object
        self.current_time = timestep_start

        # Keep only jobs that have not yet ended and that have a chance to start
        all_jobs[:] = [job for job in all_jobs if job.end_time >= timestep_start and job.submit_time < timestep_end]

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
        if replay and len(self.queue) != 0:
            raise ValueError(f"Something went wrong! Not all jobs could be placed!\nPotential confligt in queue:\n{self.queue}")
        # Restore the target policy and backfill for the remainder of the simulation.
        self.scheduler.policy = target_policy
        self.scheduler.bfpolicy = target_bfpolicy

    def run_simulation(self, jobs, timestep_start, timestep_end, time_delta=1, autoshutdown=False):
        """Generator that yields after each simulation tick."""
        self.timesteps = timestep_end - timestep_start  # Where is this used?

        if self.scheduler.policy == PolicyType.REPLAY:
            replay = True
        else:
            replay = False

        # Place jobs that are currently running, onto the system.
        self.prepare_system_state(jobs, timestep_start, timestep_end, replay)

        # Process jobs in batches for better performance of timestep loop
        all_jobs = jobs.copy()
        jobs = []
        # Batch Jobs into 6h windows based on submit_time or twice the time_delta if larger
        batch_window = max(60 * 60 * 6, 2 * time_delta)  # at least 6h

        for timestep in range(timestep_start, timestep_end):  # Runs every seconds!

            if (timestep % batch_window == 0) or (timestep == timestep_start):
                # Add jobs that are within the batching window and remove them from all jobs
                jobs += [job for job in all_jobs if job.submit_time <= timestep + batch_window]
                all_jobs[:] = [job for job in all_jobs if job.submit_time > timestep + batch_window]

            # 1. Prepare Timestep:
            completed_jobs, newly_downed_nodes = self.prepare_timestep(replay)

            # 2. Identify eligible jobs and add them to the queue.
            has_new_additions = self.add_eligible_jobs_to_queue(jobs)

            # 3. Schedule jobs that are now in the queue.
            if completed_jobs != [] or newly_downed_nodes != [] or has_new_additions:
                self.scheduler.schedule(self.queue, self.running,
                                        self.current_time,
                                        accounts=self.accounts,
                                        sorted=(not has_new_additions))

            if self.debug and timestep % self.config['UI_UPDATE_FREQ'] == 0:
                print(".", end="", flush=True)

            # 4. Run tick only at specified time_delta
            if 0 == (timestep % time_delta) and \
               ((time_delta == 1 and self.current_time % self.config['POWER_UPDATE_FREQ'] == 0) or time_delta != 1):
                tick_data = self.tick(time_delta=time_delta)
                tick_data.completed = completed_jobs
            else:
                tick_data = None

            # 5. Complete the timestep
            simulation_done = self.complete_timestep(autoshutdown, all_jobs, jobs)
            if simulation_done:
                break
            yield tick_data

    def get_stats(self):
        """ Return output statistics """
        sum_values = lambda values: sum(x[1] for x in values) if values else 0
        min_value = lambda values: min(x[1] for x in values) if values else 0
        max_value = lambda values: max(x[1] for x in values) if values else 0
        num_samples = len(self.power_manager.history) if self.power_manager else 0

        throughput = self.jobs_completed / self.timesteps * 3600 if self.timesteps else 0  # Jobs per hour
        average_power_mw = sum_values(self.power_manager.history) / num_samples / 1000 if num_samples else 0
        average_loss_mw = sum_values(self.power_manager.loss_history) / num_samples / 1000 if num_samples else 0
        min_loss_mw = min_value(self.power_manager.loss_history) / 1000 if num_samples else 0
        max_loss_mw = max_value(self.power_manager.loss_history) / 1000 if num_samples else 0

        loss_fraction = average_loss_mw / average_power_mw if average_power_mw else 0
        efficiency = 1 - loss_fraction if loss_fraction else 0
        total_energy_consumed = average_power_mw * self.timesteps / 3600 if self.timesteps else 0  # MW-hr
        emissions = total_energy_consumed * 852.3 / 2204.6 / efficiency if efficiency else 0
        total_cost = total_energy_consumed * 1000 * self.config.get('POWER_COST', 0)  # Total cost in dollars

        stats = {
            'num_samples': num_samples,
            'jobs completed': self.jobs_completed,
            'throughput': f'{throughput:.2f} jobs/hour',
            'jobs still running': [job.id for job in self.running],
            'jobs still in queue': [job.id for job in self.queue],
            'average power': f'{average_power_mw:.2f} MW',
            'min loss': f'{min_loss_mw:.2f} MW',
            'average loss': f'{average_loss_mw:.2f} MW',
            'max loss': f'{max_loss_mw:.2f} MW',
            'system power efficiency': f'{efficiency * 100:.2f}%',
            'total energy consumed': f'{total_energy_consumed:.2f} MW-hr',
            'carbon emissions': f'{emissions:.2f} metric tons CO2',
            'total cost': f'${total_cost:.2f}'
        }

        network_stats = get_network_stats()
        stats.update(network_stats)

        if self.net_util_history:
            mean_net_util = sum(self.net_util_history) / len(self.net_util_history)
        else:
            mean_net_util = 0.0
        stats["avg network util"] = f"{mean_net_util * 100:.2f}%"

        if self.avg_slowdown_history:
            avg_job_slow = sum(self.avg_slowdown_history) / len(self.avg_slowdown_history)
        else:
            avg_job_slow = 1.0
        stats["avg per-job slowdown"] = f"{avg_job_slow:.2f}x"

        if self.max_slowdown_history:
            max_job_slow = max(self.max_slowdown_history)
        else:
            max_job_slow = 1.0
        stats["max per-job slowdown"] = f"{max_job_slow:.2f}x"

        return stats

    def get_job_history_dict(self):
        return self.job_history_dict

    def get_scheduler_queue_history(self):
        return self.scheduler_queue_history

    def get_scheduler_running_history(self):
        return self.scheduler_running_history

    def record_util_stats(self,*, system_util):
        self.sys_util_history.append((self.current_time, system_util))
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
        if (time_delta == 1 and self.current_time % self.config['POWER_UPDATE_FREQ'] == 0) or time_delta != 1:
            # First job specific
            record_power_stats_foreach_job(running_jobs=self.running, jobs_power=jobs_power)
            # power manager
            self.power_manager.history.append((self.current_time, total_power_kw))
            self.power_manager.loss_history.append((self.current_time, total_loss_kw))
        #engine
        self.sys_power = total_power_kw
