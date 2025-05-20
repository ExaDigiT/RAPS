from typing import Optional, List
import dataclasses
import pandas as pd
import numpy as np

from .job import Job, JobState
from .policy import PolicyType
from .network import network_utilization, get_current_bandwidth_usage, network_dilation_factor
from .utils import summarize_ranges, expand_ranges, get_utilization
from .utils import sum_values, min_value, max_value
from .resmgr import ResourceManager
from .schedulers import load_scheduler


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
        self.sys_util_history = []
        self.scheduler_queue_history = []
        self.scheduler_running_history = []
        self.avg_net_tx = []
        self.avg_net_rx = []
        self.avg_net_util = []

        # Get scheduler type from command-line args or default
        scheduler_type = kwargs.get('scheduler', 'default')
        policy_type = kwargs.get('policy')
        backfill_type = kwargs.get('backfill')

        self.scheduler = load_scheduler(scheduler_type)(
            config=self.config,
            policy=kwargs.get('policy'),
            bfpolicy=kwargs.get('backfill'),
            resource_manager=self.resource_manager,
            jobs=jobs
        )
        print(f"Using scheduler: {str(self.scheduler.__class__).split('.')[2]}"\
              f", with policy {self.scheduler.policy} "\
              f"and backfill {self.scheduler.bfpolicy}")


    def add_running_jobs_to_queue(self, jobs_to_submit: List):
        """
        Mofifies jobs_to_submit
        and self.queue

        This is a preparatory step and should only be called before the main
        loop of run_simulation.
        Adds running jobs to the queueu, and removes them from the jobs_to_submit
        jobs_to_submit still holds the jobs that need be submitted in the future.
        """
        # Build a list of jobs whose start_time is <= current_time.
        eligible = [job for job in jobs_to_submit if job['start_time'] < self.current_time]
        # Remove those jobs from jobs_to_submit:
        jobs_to_submit[:] = [job for job in jobs_to_submit if job['start_time'] >= self.current_time]
        # Convert them to Job instances and build list of eligible jobs.
        eligible_jobs_list = []
        for job_data in eligible:
            job_instance = Job(job_data)
            eligible_jobs_list.append(job_instance)
        self.queue += eligible_jobs_list

    def add_eligible_jobs_to_queue(self, jobs_to_submit: List):
        """
        Mofifies jobs_to_submit
        and self.queue

        Adds eligible jobs to the queueu, and removes them from the jobs_to_submit
        jobs_to_submit still holds the jobs that need be submitted in the future.
        """
        # Build a list of jobs whose submit_time is <= current_time.
        eligible = [job for job in jobs_to_submit if job['submit_time'] <= self.current_time]
        # Remove those jobs from jobs_to_submit:
        jobs_to_submit[:] = [job for job in jobs_to_submit if job['submit_time'] > self.current_time]
        # Convert them to Job instances and build list of eligible jobs.
        eligible_jobs_list = []
        for job_data in eligible:
            job_instance = Job(job_data)
            eligible_jobs_list.append(job_instance)
        self.queue += eligible_jobs_list
        if eligible_jobs_list != []:
            return True
        else:
            return False

    def prepare_timestep(self, replay:bool = True):
        completed_jobs = [job for job in self.running if job.end_time is not None and job.end_time <= self.current_time]

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


    def tick(self):
        """Simulate a timestep."""

        # Update running time for all running jobs
        scheduled_nodes = []
        cpu_utils = []
        gpu_utils = []
        net_utils = []
        net_tx_list = []
        net_rx_list = []
        if self.debug:
                print(f"Current Time: {self.current_time}")
        for job in self.running:
            if job.end_time == self.current_time:
                job.state = JobState.COMPLETED

        for job in self.running:

            if self.debug:
                print(f"JobID: {job.id}")
            if job.state == JobState.RUNNING:
                job.running_time = self.current_time - job.start_time

                if job.running_time > job.wall_time:
                    raise Exception(f"Job should have ended already!\n\
                                       {job.running_time} > {job.wall_time}\n\
                                       {len(job.cpu_trace)} vs. {job.running_time // self.config['TRACE_QUANTA']}\
                                    ")

                time_quanta_index = int((job.running_time - job.trace_start_time) // self.config['TRACE_QUANTA'])
                # If the running time is past the last time step in the
                # trace, use the last value in the trace. This can
                # happen if the last valid timesteps is e.g. 17%15,
                # the last trace value is 15%15 and the next possible
                # trace value 30%15 but was not recorded because the
                # job ended before.
                # For every other error condition trace_start_ and
                # _end_time are used!
                # #print(type(job.cpu_trace))
                if time_quanta_index < 0:
                    time_quanta_index = 0
                # Similar with the first time_quanta index: If the job started
                # in the past and no trace if there, read index 0 until values
                # are available.
                if isinstance(job.cpu_trace,list) or isinstance(job.cpu_trace,np.ndarray):
                    if time_quanta_index < len(job.cpu_trace):
                        cpu_util = get_utilization(job.cpu_trace, time_quanta_index)
                    else:
                        cpu_util = get_utilization(job.cpu_trace, len(job.cpu_trace) - 1)
                elif isinstance(job.cpu_trace,float) or isinstance(job.cpu_trace,int):
                    cpu_util = job.cpu_trace
                else:
                    raise NotImplementedError()

                if isinstance(job.gpu_trace,list) or isinstance(job.gpu_trace,np.ndarray):
                    if time_quanta_index < len(job.gpu_trace):
                        gpu_util = get_utilization(job.gpu_trace, time_quanta_index)
                    else:
                        gpu_util = get_utilization(job.gpu_trace, len(job.gpu_trace) - 1)
                elif isinstance(job.gpu_trace,float) or isinstance(job.gpu_trace,int):
                    gpu_util = job.gpu_trace
                else:
                    raise NotImplementedError()

                net_util = 0

                if (isinstance(job.ntx_trace,list) or isinstance(job.ntx_trace,np.ndarray)) and len(job.ntx_trace) and (isinstance(job.nrx_trace,list) or isinstance(job.nrx_trace,list)) and len(job.nrx_trace):
                    max_link_bw = self.config.get('NETWORK_MAX_BW')
                    net_tx = get_utilization(job.ntx_trace, time_quanta_index)
                    net_rx = get_utilization(job.nrx_trace, time_quanta_index)
                    net_util = network_utilization(net_tx, net_rx, max_link_bw)
                    net_tx_list.append(net_tx)
                    net_rx_list.append(net_rx)
                    #net_utils.append(net_util)
                    if self.debug:
                        print("time:", self.current_time, "net util:", net_util)
                        print("jid", job.id, "net_tx", net_tx)
                        print("jid", job.id, "net_rx", net_tx)
                    net_utils.append(net_util)

                    # Get the maximum allowed bandwidth from the configuration.
                    if net_util > 1: #network_congestion_threshold:
                        if self.debug:
                            print(f"congested net_util: {net_util}, max_link_bw: {max_link_bw}")
                            print(f"length of {len(job.gpu_trace)} before dilation")
                        # Use our network helper functions to get current bandwidth usage.
                        #current_bw = get_current_bandwidth_usage(link_id="link_1")
                        current_bw = net_tx + net_rx
                        dilation_factor = network_dilation_factor(current_bw, max_link_bw)
                        dilation_factor = min(dilation_factor, 2) # set max dilation factor
                        # Optionally, only apply dilation once per job to avoid compounding the effect.
                        if self.debug:
                            print("***", hasattr(job, 'network_dilated'), current_bw, max_link_bw, dilation_factor)
                        #if not hasattr(job, 'network_dilated') or not job.network_dilated:
                        if not job.network_dilated:
                            if self.debug:
                                print(f"Applying dilation factor {dilation_factor:.2f} to job {job.id} due to network congestion")
                            job.apply_dilation(dilation_factor)
                            job.network_dilated = True
                            if self.debug:
                                print(f"length of {len(job.gpu_trace)} after dilation")

                else:
                    net_utils.append(0)

                scheduled_nodes.append(job.scheduled_nodes)  # ?
                cpu_utils.append(cpu_util)
                gpu_utils.append(gpu_util)
            else:
                raise ValueError(f"Job is in running list, but state is not RUNNING: job.state == {job.state}")

        if len(scheduled_nodes) > 0:
            self.flops_manager.update_flop_state(scheduled_nodes, cpu_utils, gpu_utils)
            jobs_power = self.power_manager.update_power_state(scheduled_nodes, cpu_utils, gpu_utils, net_utils)

            _running_jobs = [job for job in self.running if job.state == JobState.RUNNING]
            if len(jobs_power) != len(_running_jobs):
                raise ValueError(f"Jobs power list of length ({len(jobs_power)}) should have ({len(_running_jobs)}) items.")
            for i, job in enumerate(_running_jobs):
                if job.running_time % self.config['TRACE_QUANTA'] == 0:
                    job.power_history.append(jobs_power[i] * len(job.scheduled_nodes))
            del _running_jobs

        # Update the power array UI component
        rack_power, rect_losses = self.power_manager.compute_rack_power()
        sivoc_losses = self.power_manager.compute_sivoc_losses()
        rack_loss = rect_losses + sivoc_losses

        # Update system utilization
        system_util = self.num_active_nodes / self.config['AVAILABLE_NODES'] * 100
        self.sys_util_history.append((self.current_time, system_util))

        self.scheduler_queue_history.append(len(self.running))
        self.scheduler_running_history.append(len(self.queue))

        # Render the updated layout
        power_df = None
        cooling_inputs, cooling_outputs = None, None

        # Update power history every 15s
        if self.current_time % self.config['POWER_UPDATE_FREQ'] == 0:
            total_power_kw = sum(row[-1] for row in rack_power) + self.config['NUM_CDUS'] * self.config['POWER_CDU'] / 1000.0
            total_loss_kw = sum(row[-1] for row in rack_loss)
            self.power_manager.history.append((self.current_time, total_power_kw))
            self.sys_power = total_power_kw
            self.power_manager.loss_history.append((self.current_time, total_loss_kw))
            pflops = self.flops_manager.get_system_performance() / 1E15
            gflop_per_watt = pflops * 1E6 / (total_power_kw * 1000)
        else:
            pflops, gflop_per_watt = None, None

        if self.current_time % self.config['POWER_UPDATE_FREQ'] == 0:
            if self.cooling_model:
                # Power for NUM_CDUS (25 for Frontier)
                cdu_power = rack_power.T[-1] * 1000
                runtime_values = self.cooling_model.generate_runtime_values(cdu_power, self)

                # FMU inputs are N powers and the wetbulb temp
                fmu_inputs = self.cooling_model.generate_fmu_inputs(runtime_values,
                                                                    uncertainties=self.power_manager.uncertainties)
                cooling_inputs, cooling_outputs = (
                    self.cooling_model.step(self.current_time, fmu_inputs, self.config['POWER_UPDATE_FREQ'])
                )

                # Get a dataframe of the power data
                power_df = self.power_manager.get_power_df(rack_power, rack_loss)
            else:
                # Get a dataframe of the power data
                power_df = self.power_manager.get_power_df(rack_power, rack_loss)

        # Compute network averages
        n = len(net_utils) or 1
        avg_tx   = sum(net_tx_list) / n
        avg_rx   = sum(net_rx_list) / n
        avg_net  = sum(net_utils)   / n

        # Save network history
        self.avg_net_tx.append(avg_tx)
        self.avg_net_rx.append(avg_rx)
        self.avg_net_util.append(avg_net)

        tick_data = TickData(
            current_time=self.current_time,
            completed=None,
            running=self.running,
            queue=self.queue,
            down_nodes=expand_ranges(self.down_nodes[1:]),
            power_df=power_df,
            p_flops=pflops,
            g_flops_w=gflop_per_watt,
            system_util=self.num_active_nodes / self.config['AVAILABLE_NODES'] * 100,
            fmu_inputs=cooling_inputs,
            fmu_outputs=cooling_outputs,
            num_active_nodes=self.num_active_nodes,
            num_free_nodes=self.num_free_nodes,
            avg_net_tx=avg_tx,
            avg_net_rx=avg_rx,
            avg_net_util=avg_net
        )

        self.current_time += 1
        return tick_data

    def prepare_system_state(self, all_jobs:List, timestep_start, timestep_end, replay:bool):
        # Modifies Jobs object
        self.current_time = timestep_start

        # Keep only jobs that have not yet ended and that have a chance to start
        all_jobs[:] = [job for job in all_jobs if job['end_time'] >= timestep_start and job['submit_time'] < timestep_end]

        all_jobs.sort(key=lambda j: j['submit_time'])

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

    def run_simulation(self, jobs, timestep_start, timestep_end, autoshutdown=False):
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
        # Batch Jobs into 6h windows based on submit_time
        batch_window = 60 * 60 * 6  # 6h

        for timestep in range(timestep_start,timestep_end):

            if (timestep % batch_window == 0) or (timestep == timestep_start):
                # Add jobs that are within the batching window and remove them from all jobs
                jobs += [job for job in all_jobs if job['submit_time'] <= timestep + batch_window]
                all_jobs[:] = [job for job in all_jobs if job['submit_time'] > timestep + batch_window]

            # Start Siulation loop:
            # 1. Cleanup old jobs
            completed_jobs, newly_downed_nodes = self.prepare_timestep(replay)

            # 2. Identify eligible jobs and add them to the queue.
            has_new_additions = self.add_eligible_jobs_to_queue(jobs)
            # 3. Schedule jobs that are now in the queue.
            self.scheduler.schedule(self.queue, self.running, self.current_time,accounts=self.accounts, sorted=(not has_new_additions))

            # Stop the simulation if no more jobs are running or in the queue or in the job list.
            if autoshutdown and not self.queue and not self.running and not self.replay and not all_jobs and not jobs:
                print(f"[DEBUG] {self.config['system_name']} - Stopping simulation at time {self.current_time}")
                break

            if self.debug and timestep % self.config['UI_UPDATE_FREQ'] == 0:
                print(".", end="", flush=True)

            tick_data = self.tick()
            tick_data.completed = completed_jobs
            yield tick_data

    def get_job_history_dict(self):
        return self.job_history_dict

    def get_scheduler_queue_history(self):
        return self.scheduler_queue_history

    def get_scheduler_running_history(self):
        return self.scheduler_running_history
