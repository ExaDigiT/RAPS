"""A module for job scheduling and simulation in a distributed computing environment.

This module provides classes and functions for managing job scheduling and simulating the behavior
of a distributed computing system. It includes functionalities for scheduling jobs based on various
policies, simulating the passage of time, handling node failures, and generating statistics about
the simulation.

Classes:
- JobState: An enumeration representing the states of a job.
- Job: A class representing a job to be scheduled and executed.
- TickData: A dataclass representing the state output from the simulation each tick.
- Scheduler: A class for job scheduling and simulation management.

Functions:
- summarize_ranges: A utility function to summarize ranges of values.
- expand_ranges: A utility function to expand ranges of values into individual elements.

Dependencies:
- numpy: For numerical computations and array manipulations.
- dataclasses: For creating data classes with less boilerplate code.
- pandas: For data manipulation and DataFrame generation.
- enum: For creating enumerations.
- scipy.stats: For statistical distributions and random number generation.
- typing: For type hints and annotations.

Constants:
- TRACE_QUANTA: The quantum of time for tracing job CPU and GPU utilization.
- MTBF: Mean Time Between Failures, used for node failure simulation.
- POWER_COST: Cost of power consumption per unit, used for calculating total cost.
- UI_UPDATE_FREQ: Frequency of updating the user interface.
- MAX_TIME: Maximum simulation time.
- POWER_UPDATE_FREQ: Frequency of updating power-related metrics.
- POWER_DF_HEADER: Header for the power related components of DataFrame.
- FMU_UPDATE_FREQ: Frequency of updating the FMU model.
- POWER_CDU: Power consumption of CDU.
- TOTAL_NODES: Total number of nodes in the system.
- COOLING_EFFICIENCY: Cooling efficiency factor.

This module can be used to simulate job scheduling algorithms, analyze system behavior, and
optimize resource utilization in distributed computing environments.
"""
from typing import Optional
import dataclasses
import numpy as np

from scipy.stats import weibull_min
import pandas as pd

from .config import load_config_variables
from .job import Job, JobState
from .policy import Policy, PolicyType
from .utils import summarize_ranges, expand_ranges

load_config_variables([
    'TRACE_QUANTA',
    'MTBF',
    'POWER_COST',
    'UI_UPDATE_FREQ',
    'MAX_TIME',
    'POWER_UPDATE_FREQ',
    'FMU_UPDATE_FREQ',
    'POWER_CDU',
    'TOTAL_NODES',
    'COOLING_EFFICIENCY',
    'WET_BULB_TEMP',
    'NUM_CDUS',
    'POWER_DF_HEADER',
    'AVAILABLE_NODES',
    'TOTAL_NODES'
], globals())


@dataclasses.dataclass
class TickData:
    """ Represents the state output from the simulation each tick """
    current_time: int
    jobs: list[Job]
    down_nodes: list[int]
    power_df: Optional[pd.DataFrame]
    p_flops: float
    g_flops_w: float
    system_util: float
    fmu_inputs: Optional[dict]
    fmu_datacenter_outputs: Optional[dict]
    fmu_cep_outputs: Optional[dict]
    pue: Optional[float]


class Scheduler:
    """Job scheduler and simulation manager."""
    def __init__(self, total_nodes, down_nodes, power_manager, flops_manager, \
                 layout_manager, cooling_model=None, **kwargs):
        """Initialize the scheduler.

        Args:
            total_nodes: Total number of nodes in the system.
            down_nodes: Nodes that are already down.
            power_manager: Instance of the PowerManager class.
            layout_manager: Instance of the LayoutManager class.
            cooling_model: Cooling model for the system.
            debug: Flag for enabling debug mode.
            output: Flag for enabling output.

        Attributes:
            total_nodes: Total number of nodes in the system.
            available_nodes: Nodes available for scheduling.
            down_nodes: Nodes that are down.
            num_free_nodes: Number of free nodes.
            num_active_nodes: Number of active nodes.
            debug: Flag for debug mode.
            running: List of running jobs.
            queue: Queue of pending jobs.
            jobs_completed: Number of jobs completed.
            current_time: Current simulation time.
            cooling_model: Cooling model for the system.
            layout_manager: Layout manager for rendering.
            power_manager: Power manager instance.
            fmu_results: Results from the FMU model.
            output: Flag for enabling output.
        """
        self.total_nodes = total_nodes
        self.available_nodes = list(set(range(total_nodes)) - set(down_nodes))
        self.down_nodes = summarize_ranges(down_nodes)
        self.num_free_nodes = len(self.available_nodes)
        self.num_active_nodes = TOTAL_NODES - self.num_free_nodes - len(expand_ranges(self.down_nodes))
        self.running = []
        self.queue = []
        self.jobs_completed = 0
        self.current_time = 0
        self.cooling_model = cooling_model
        self.layout_manager = layout_manager
        self.power_manager = power_manager
        self.flops_manager = flops_manager
        self.debug = kwargs.get('debug')
        self.output = kwargs.get('output')
        self.replay = kwargs.get('replay')
        self.policy = Policy(strategy=kwargs.get('schedule'))
        self.sys_util_history = []


    def add_job(self, job):
        self.queue.append(job)
        self.queue = self.policy.sort_jobs(self.queue)


    def assign_nodes_to_job(self, job):
        """Helper function to assign nodes to a job and update available nodes."""
        if len(self.available_nodes) < job.nodes_required:
            # If there are not enough nodes, return or raise an error (handle as needed)
            raise ValueError(f"Not enough available nodes to schedule job {job.id}")

        if job.requested_nodes: # replay case
            # If the job has requested specific nodes, assign them
            job.scheduled_nodes = job.requested_nodes
            mask = ~np.isin(self.available_nodes, job.scheduled_nodes)
            self.available_nodes = np.array(self.available_nodes)
            self.available_nodes = self.available_nodes[mask]
            self.available_nodes = self.available_nodes.tolist()
        else: # synthetic or reschedule case
            # Assign the nodes from available pool
            job.scheduled_nodes = self.available_nodes[:job.nodes_required]
            self.available_nodes = self.available_nodes[job.nodes_required:]

        # Set job start and end times
        job.start_time = self.current_time
        job.end_time = self.current_time + job.wall_time

        # Mark the job as running
        job.state = JobState.RUNNING
        self.running.append(job)


    def schedule(self, jobs):
        """Schedule jobs"""
        for job_info in jobs:
            job = Job(job_info, self.current_time)
            self.add_job(job)

        while self.queue:

            # Try scheduling the first job in the queue
            job = self.queue.pop(0)
            synthetic_bool = len(self.available_nodes) >= job.nodes_required
            telemetry_bool = job.requested_nodes and job.requested_nodes[0] in self.available_nodes

            if synthetic_bool or telemetry_bool:

                # Schedule job
                self.assign_nodes_to_job(job)

                if self.debug:
                    scheduled_nodes = summarize_ranges(job.scheduled_nodes)
                    print(f"t={self.current_time}: Scheduled job with wall time",
                          f"{job.wall_time} on nodes {scheduled_nodes}")

            else:
                # If the job cannot be scheduled, either try backfilling or requeue it
                if self.queue and self.policy.strategy == PolicyType.BACKFILL:
                    self.queue.insert(0, job)
                    backfill_job = self.policy.find_backfill_job(self.queue, len(self.available_nodes), self.current_time)
                    if backfill_job:
                        self.assign_nodes_to_job(backfill_job)
                        self.queue.remove(backfill_job)
                        if self.debug:
                            scheduled_nodes = summarize_ranges(backfill_job.scheduled_nodes)
                            print(f"t={self.current_time}: Backfilling job {backfill_job.id} with wall time",
                                  f"{backfill_job.wall_time} on nodes {scheduled_nodes}")
                else:
                    self.queue.append(job)
                break


    def tick(self):
        """Simulate a timestep."""
        completed_jobs = [job for job in self.running if job.end_time
                          is not None and job.end_time <= self.current_time]

        # Simulate node failure
        newly_downed_nodes = self.node_failure(MTBF)

        # Update active/free nodes
        self.num_free_nodes = len(self.available_nodes)
        self.num_active_nodes = TOTAL_NODES - self.num_free_nodes \
                              - len(expand_ranges(self.down_nodes))

        # Update running time for all running jobs
        for job in self.running:

            if job.end_time == self.current_time:
                job.state = JobState.COMPLETED

            if job.state == JobState.RUNNING:
                # Deal with node that fails during the course of a running job
                #if any(node in job.scheduled_nodes for node in newly_downed_nodes):
                if False: # currently disabled b/c not working correctly

                    # Update job state to FAILED
                    job.state = JobState.FAILED

                    # Release all nodes except the downed node
                    for node in job.scheduled_nodes:
                        if node not in newly_downed_nodes:
                            self.available_nodes.append(node)
                    self.available_nodes.sort()

                    # Remove job from the list of running jobs
                    self.running.remove(job)

                job.running_time = self.current_time - job.start_time

                time_quanta_index = (self.current_time - job.start_time) // TRACE_QUANTA
                cpu_util = job.cpu_trace[time_quanta_index]
                gpu_util = job.gpu_trace[time_quanta_index]

                self.flops_manager.update_flop_state(job.scheduled_nodes, cpu_util, gpu_util)
                job.power = self.power_manager.update_power_state(job.scheduled_nodes,
                                                                  cpu_util, gpu_util)

                if job.running_time % TRACE_QUANTA == 0:
                    job.power_history.append(job.power)

        for job in completed_jobs:
            # Release the nodes used by this job
            self.available_nodes.extend(job.scheduled_nodes)
            self.available_nodes.sort()

            if self.debug:
                print(
                    f"\nt={self.current_time}: "
                    f"Releasing {len(job.scheduled_nodes)} nodes from completed job; "
                    f"{len(self.available_nodes)} nodes available after release."
                )

            # Set nodes back to idle power
            node_indices = np.array(job.scheduled_nodes)
            if self.debug:
                print("setting idle nodes:", node_indices)
            self.power_manager.set_idle(node_indices)
            self.flops_manager.update_flop_state(job.scheduled_nodes, \
                                                 cpu_util=0, gpu_util=0)

            # Remove job from list of running jobs
            self.running.remove(job)
            scheduled_nodes = summarize_ranges(job.scheduled_nodes)

            if self.debug:
                print(f"Released {scheduled_nodes}")
            self.jobs_completed += 1

            if self.output:
                with open(self.opath / f'job-power-{job.id}.txt', 'w') as file:
                    print(*job.power_history, sep=', ', file=file)

        # Ask scheduler to schedule any jobs waiting in queue
        self.schedule([])

        # Update the power array UI component
        rack_power, rect_losses = self.power_manager.compute_rack_power()
        sivoc_losses = self.power_manager.compute_sivoc_losses()
        rack_loss = rect_losses + sivoc_losses

        # Update system utilization
        system_util = self.num_active_nodes / AVAILABLE_NODES * 100
        self.sys_util_history.append((self.current_time, system_util))

        # Render the updated layout
        power_df = None
        cooling_inputs, datacenter_outputs, cep_outputs, pue = None, None, None, None

        # Update power history every 15s
        if self.current_time % POWER_UPDATE_FREQ == 0:
            total_power_kw = sum(row[-1] for row in rack_power) + NUM_CDUS * POWER_CDU / 1000.0
            total_loss_kw = sum(row[-1] for row in rack_loss)
            self.power_manager.history.append((self.current_time, total_power_kw))
            self.power_manager.loss_history.append((self.current_time, total_loss_kw))
            output_df = self.power_manager.get_power_df(rack_power, rack_loss)
            pflops = self.flops_manager.get_system_performance() / 1E15
            gflop_per_watt = pflops * 1E6 / (total_power_kw * 1000)
        else:    
            pflops, gflop_per_watt = None, None

        if self.cooling_model:

            if self.current_time % FMU_UPDATE_FREQ == 0:
                # Power for NUM_CDUS (25 for Frontier)
                cdu_power = rack_power.T[-1] * 1000
                runtime_values = self.cooling_model.generate_runtime_values(cdu_power, self)
                
                # FMU inputs are N powers and the wetbulb temp
                fmu_inputs = self.cooling_model.generate_fmu_inputs(runtime_values, \
                             uncertainties=self.power_manager.uncertainties)
                cooling_inputs, datacenter_outputs, cep_outputs, pue = \
                    self.cooling_model.step(self.current_time, fmu_inputs, FMU_UPDATE_FREQ)
                
                # Get a dataframe of the power data
                power_df = self.power_manager.get_power_df(rack_power, rack_loss)

                if self.layout_manager:
                    self.layout_manager.update_powertemp_array(power_df, \
                               datacenter_outputs, pue, pflops, gflop_per_watt, \
                               system_util, uncertainties=self.power_manager.uncertainties)
                    self.layout_manager.update_pressflow_array(datacenter_outputs)

        if self.current_time % UI_UPDATE_FREQ == 0:
            # Get a dataframe of the power data
            power_df = self.power_manager.get_power_df(rack_power, rack_loss)

            if self.layout_manager and not self.debug:
                self.layout_manager.update_scheduled_jobs(self.running + self.queue)
                self.layout_manager.update_status(self.current_time, len(self.running),
                                              len(self.queue), self.num_active_nodes,
                                              self.num_free_nodes, self.down_nodes[1:])
                self.layout_manager.update_power_array(power_df, pflops, gflop_per_watt, \
                                    system_util, uncertainties=self.power_manager.uncertainties)
                self.layout_manager.render()

        tick_data = TickData(
            current_time = self.current_time,
            jobs = completed_jobs + self.running + self.queue,
            down_nodes = expand_ranges(self.down_nodes[1:]),
            power_df = power_df,
            p_flops = pflops,
            g_flops_w = gflop_per_watt,
            system_util = system_util,
            fmu_inputs = cooling_inputs,
            fmu_datacenter_outputs = datacenter_outputs,
            fmu_cep_outputs = cep_outputs,
            pue = pue
        )

        self.current_time += 1
        return tick_data

    def get_gauge_limits(self):
        """For setting max values in dashboard gauges"""
        peak_flops = self.flops_manager.get_rpeak()
        peak_power = self.power_manager.get_peak_power()
        gflops_per_watt_max = peak_flops / 1E9 / peak_power

        if self.debug:
            print(f"System Rpeak: {peak_flops/1E15:.2f} PFLOPS")
            print(f"Peak power: {peak_power/1E3:.0f} kW")
            print(f"Max energy efficiency: {gflops_per_watt_max:.1f} GFLOPS/W")

        limits = {'peak_flops': peak_flops, 'peak_power': peak_power, \
                  'g_flops_w_peak': gflops_per_watt_max}
        return limits

    def run_simulation(self, jobs, timesteps):
        """ Generator that yields after each simulation tick """
        time_to_next_job = 0
        self.timesteps = timesteps
        if self.debug: 
            limits = self.get_gauge_limits()
            print(limits)
        
        for _ in range(timesteps):
            while self.current_time >= time_to_next_job and jobs:
                job = jobs.pop(0)
                self.schedule([job])
                if jobs: # Update time to next job based on the next job's scheduled time
                    time_to_next_job = job['submit_time']
                else: # No more jobs, set to infinity or some large number to avoid triggering again
                    time_to_next_job = float('inf')
            yield self.tick()

            # Stop the simulation if no more jobs running or are in the queue
            if not self.queue and not self.running and not self.replay:
                print("stopping simulation at time", self.current_time)
                break
            if self.debug:
                if _ % UI_UPDATE_FREQ == 0:
                    print(".", end="", flush=True)

    def run_simulation_blocking(self, jobs, timesteps):
        """ Calls run_simulation and blocks until it is complete """
        for _ in self.run_simulation(jobs, timesteps):
            pass

    def get_stats(self):
        """ Return output statistics """
        sum_values = lambda values : sum(x[1] for x in values)
        min_value = lambda values : min(x[1] for x in values)
        max_value = lambda values : max(x[1] for x in values)
        num_samples = len(self.power_manager.history)
        throughput = self.jobs_completed / self.timesteps * 3600 # jobs/hour
        average_power_mw = sum_values(self.power_manager.history) / num_samples / 1000
        average_loss_mw = sum_values(self.power_manager.loss_history) / num_samples / 1000
        min_loss_mw = min_value(self.power_manager.loss_history) / 1000
        max_loss_mw = max_value(self.power_manager.loss_history) / 1000
        self.power_manager.loss_history_percentage = \
            [(x[0], x[1] / y[1]) for x, y in zip(self.power_manager.loss_history, \
                                                 self.power_manager.history)]
        min_loss_pct = min_value(self.power_manager.loss_history_percentage)
        max_loss_pct = max_value(self.power_manager.loss_history_percentage)

        loss_fraction = average_loss_mw / average_power_mw
        efficiency = 1 - loss_fraction
        # compute total power consumed by multiplying average power times length of simulation
        total_energy_consumed = average_power_mw * self.timesteps / 3600 # MW-hr
        # From https://www.epa.gov/energy/greenhouse-gases-equivalencies-\
        #      calculator-calculations-and-references
        emissions = total_energy_consumed * 852.3 / 2204.6 / efficiency
        total_cost = total_energy_consumed * 1000 * POWER_COST # total cost in dollars

        stats = {
            'num_samples': num_samples,
            'jobs completed': self.jobs_completed,
            'throughput': f'{throughput:.2f} jobs/hour',
            'jobs still running': [job.id for job in self.running],
            'jobs still in queue': [job.id for job in self.queue],
            'average power': f'{average_power_mw:.2f} MW',
            'min loss': f'{min_loss_mw:.2f} MW ({min_loss_pct*100:.2f}%)', 
            'average loss': f'{average_loss_mw:.2f} MW ({loss_fraction*100:.2f}%)',
            'max loss': f'{max_loss_mw:.2f} MW ({max_loss_pct*100:.2f}%)', 
            'system power efficiency': f'{efficiency*100:.2f}',
            'total energy consumed': f'{total_energy_consumed:.2f} MW-hr',
            'carbon emissions': f'{emissions:.2f} metric tons CO2',
            'total cost': f'${total_cost:.2f}'
        }

        return stats

    def node_failure(self, mtbf):
        """Simulate node failure."""
        shape_parameter = 1.5
        scale_parameter = mtbf * 3600 # to seconds

        # Create a NumPy array of node indices, excluding down nodes
        down_nodes = expand_ranges(self.down_nodes)
        all_nodes = np.setdiff1d(np.arange(self.total_nodes), np.array(down_nodes, dtype=int))

        # Sample the Weibull distribution for all nodes at once
        random_values = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=all_nodes.size)

        # Identify nodes that have failed
        failure_threshold = 0.1
        failed_nodes_mask = random_values < failure_threshold
        newly_downed_nodes = all_nodes[failed_nodes_mask]

        # Update available and down nodes
        for node_index in newly_downed_nodes:
            if node_index in self.available_nodes:
                self.available_nodes.remove(node_index)
            self.down_nodes.append(str(node_index))
            self.power_manager.set_idle(node_index)

        return newly_downed_nodes.tolist()
