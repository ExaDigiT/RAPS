"""
Module Description:
This module contains functions and classes related to workload management and power calculation.

Classes:
- PowerManager: Manages power consumption and loss calculations in the system.

Functions:
- compute_loss: Linear loss model
- compute_node_power: Calculate the total power consumption for given CPU and GPU utilization.
- compute_node_power_validate: Calculate the total power consumption for
    a given mean and standard deviation of node power.
"""

import numpy as np
import pandas as pd
import uncertainties as uf
from .utils import linear_to_3d_index


def custom_str_uncertainties(self):
    return f"{self.nominal_value} ± {self.std_dev}"


def custom_repr_uncertainties(self):
    return f"{self.nominal_value}+/-{self.std_dev}"


def custom_format_uncertainties(self, fmt_spec):
    return f"{self.nominal_value:{fmt_spec}} ±{self.std_dev:{fmt_spec}}"


# In stats unicode is printed as unocde abbreviation! To be fixed!
uf.Variable.__str__ = custom_str_uncertainties
uf.Variable.__repr__ = custom_repr_uncertainties
uf.Variable.__format__ = custom_format_uncertainties


def compute_loss(p_out, loss_constant, efficiency):
    return (p_out + loss_constant) / efficiency


def compute_node_power(cpu_util, gpu_util, net_util, config):
    """
    Calculate the total power consumption for given CPU and GPU utilization.

    :param cpu_util: The utilization of the CPU.
    :param gpu_util: The utilization of the GPU.
    :param verbose: Flag for verbose output.
    :return: Total power consumption after accounting for power loss.
    """
    power_cpu = cpu_util * config['POWER_CPU_MAX'] + \
        (config['CPUS_PER_NODE'] - cpu_util) * config['POWER_CPU_IDLE']

    power_gpu = gpu_util * config['POWER_GPU_MAX'] + \
        (config['GPUS_PER_NODE'] - gpu_util) * config['POWER_GPU_IDLE']

    if config.get("POWER_NIC_IDLE") != None and config.get("POWER_NIC_MAX") != None:
        power_nic = config['POWER_NIC_IDLE'] + \
            (config['POWER_NIC_MAX'] - config['POWER_NIC_IDLE']) * net_util
    else:
        if isinstance(net_util, np.ndarray):
            power_nic = config['POWER_NIC'] * np.ones(net_util.shape)
        else:
            power_nic = config['POWER_NIC']

    power_total = power_cpu + power_gpu + config['POWER_MEM'] + \
        config['NICS_PER_NODE'] * power_nic + config['POWER_NVME']

    # Apply power loss due to Sivoc and Rectifier
    power_with_sivoc_loss = compute_loss(power_total, config['SIVOC_LOSS_CONSTANT'],
                                         config['SIVOC_EFFICIENCY'])
    power_sivoc_loss_only = power_with_sivoc_loss - power_total

    return power_with_sivoc_loss, power_sivoc_loss_only


def compute_node_power_uncertainties(cpu_util, gpu_util, net_util, config):
    """
    Calculate the total power consumption for given CPU and GPU utilization.

    :param cpu_util: The utilization of the CPU.
    :param gpu_util: The utilization of the GPU.
    :param verbose: Flag for verbose output.
    :return: Total power consumption after accounting for power loss.
    """
    power_cpu = cpu_util \
        * uf.ufloat(config['POWER_CPU_MAX'], config['POWER_CPU_MAX'] * config['POWER_CPU_UNCERTAINTY']) \
        + (config['CPUS_PER_NODE'] - cpu_util) \
        * uf.ufloat(config['POWER_CPU_IDLE'], config['POWER_CPU_IDLE'] * config['POWER_CPU_UNCERTAINTY'])
    power_gpu = gpu_util \
        * uf.ufloat(config['POWER_GPU_MAX'], config['POWER_GPU_MAX'] * config['POWER_GPU_UNCERTAINTY']) \
        + (config['GPUS_PER_NODE'] - gpu_util) \
        * uf.ufloat(config['POWER_GPU_IDLE'], config['POWER_GPU_IDLE'] * config['POWER_GPU_UNCERTAINTY'])

    power_total = power_cpu + power_gpu \
        + uf.ufloat(config['POWER_MEM'], config['POWER_MEM'] * config['POWER_MEM_UNCERTAINTY']) \
        + config['NICS_PER_NODE'] \
        * uf.ufloat(config['POWER_NIC'], config['POWER_NIC'] * config['POWER_NIC_UNCERTAINTY']) \
        + uf.ufloat(config['POWER_NVME'], config['POWER_NVME'] * config['POWER_NVME_UNCERTAINTY'])

    # Apply power loss due to Sivoc and Rectifier
    power_with_sivoc_loss = compute_loss(power_total, config['SIVOC_LOSS_CONSTANT'], config['SIVOC_EFFICIENCY'])
    power_sivoc_loss_only = power_with_sivoc_loss - power_total

    return power_with_sivoc_loss, power_sivoc_loss_only


def compute_node_power_validate(mean_node_power, stddev_node_power, net_util, config):
    """
    Calculate the total power consumption for given mean and standard deviation of node power.

    Parameters:
    - mean_node_power: float
        Mean node power consumption.
    - stddev_node_power: float
        Standard deviation of node power consumption.
    - verbose: bool, optional
        Flag for verbose output. Default is False.

    Returns:
    tuple
        Total power consumption after accounting for power loss and Sivoc loss.
    """
    power_total = mean_node_power
    power_with_sivoc_loss = compute_loss(power_total, config['SIVOC_LOSS_CONSTANT'], config['SIVOC_EFFICIENCY'])
    power_sivoc_loss_only = power_with_sivoc_loss - power_total
    return power_with_sivoc_loss, power_sivoc_loss_only


def compute_node_power_validate_uncertainties(mean_node_power, stddev_node_power, net_util, config):
    """
    Calculate the total power consumption for given mean and standard deviation of node power.

    Parameters:
    - mean_node_power: float
        Mean node power consumption.
    - stddev_node_power: float
        Standard deviation of node power consumption.
    - verbose: bool, optional
        Flag for verbose output. Default is False.

    Returns:
    tuple
        Total power consumption after accounting for power loss and Sivoc loss.
    """
    power_total = uf.ufloat(mean_node_power, mean_node_power * config['POWER_NODE_UNCERTAINTY'])
    power_with_sivoc_loss = compute_loss(power_total, config['SIVOC_LOSS_CONSTANT'], config['SIVOC_EFFICIENCY'])
    power_sivoc_loss_only = power_with_sivoc_loss - power_total
    return power_with_sivoc_loss, power_sivoc_loss_only


class PowerManager:
    """
    Class Description:
    Manages power consumption and loss calculations in the system.

    Attributes:
    - sc_shape: Shape of the system configuration.
    - power_func: Function for calculating power consumption.
    - power_state: Current power state of the system.
    - rectifier_loss: Loss due to rectifier inefficiency.
    - sivoc_loss: Loss due to Sivoc inefficiency.
    - history: History of power states.
    - loss_history: History of power losses.
    - down_nodes: Nodes that are currently down.
    - down_rack: Rack number of down nodes.
    """

    def __init__(self, power_func=compute_node_power, **config):
        """
        Initialize the PowerManager object.

        Parameters:
        - sc_shape: tuple
            Shape of the system configuration.
        - down_nodes: list
            Nodes that are currently down.
        - power_func: function, optional
            Function for calculating power consumption. Default is compute_node_power.
        """
        self.sc_shape = config.get('SC_SHAPE')
        self.down_nodes = config.get('DOWN_NODES')
        self.config = config
        self.power_func = power_func
        self.power_state = self.initialize_power_state()
        self.rectifier_loss = self.initialize_rectifier_loss()
        self.sivoc_loss = self.initialize_sivoc_loss()
        self.history = []
        self.loss_history = []
        self.uncertainties = False
        if power_func in [compute_node_power_uncertainties,
                          compute_node_power_validate_uncertainties]:
            self.uncertainties = True
        if self.down_nodes:
            self.apply_down_nodes()

    def get_peak_power(self):
        """Estimate peak power of system for setting max value of gauges in dashboard"""
        node_power = compute_node_power(self.config['CPUS_PER_NODE'],
                                        self.config['GPUS_PER_NODE'], net_util=0, config=self.config)[0]
        blades_per_rectifier = self.config['BLADES_PER_CHASSIS'] / self.config['RECTIFIERS_PER_CHASSIS']
        rectifier_load = blades_per_rectifier * self.config['NODES_PER_BLADE'] * node_power
        rectifier_power = compute_loss(rectifier_load, self.config['RECTIFIER_LOSS_CONSTANT'],
                                       self.config['RECTIFIER_EFFICIENCY'])  # with AC-DC conversion losses
        chassis_power = self.config['BLADES_PER_CHASSIS'] * rectifier_power / blades_per_rectifier \
            + self.config['SWITCHES_PER_CHASSIS'] * self.config['POWER_SWITCH']
        rack_power = chassis_power * self.config['CHASSIS_PER_RACK']
        total_power = rack_power * self.config['NUM_RACKS'] + self.config['POWER_CDU'] * self.config['NUM_CDUS']
        return total_power

    def initialize_power_state(self):
        """Initialize the power state array with idle power consumption values."""
        initial_power, _ = self.power_func(0, 0, 0, self.config)
        return np.full(self.sc_shape, initial_power)

    def initialize_sivoc_loss(self):
        """Initialize the Sivoc loss array with idle power consumption values."""
        _, initial_sivoc_loss = self.power_func(0, 0, 0, self.config)
        return np.full(self.sc_shape, initial_sivoc_loss)

    def initialize_rectifier_loss(self):
        """ Initialize the power state array """
        initial_power, _ = self.power_func(0, 0, 0, self.config)
        # Rectifier loss curvefit is done at rectifier level, so we simply
        # approximate by scaling up to number of rectifiers, applying loss
        # and then dividing by number of rectifiers.
        # For Frontier there are four nodes per rectifier.
        power_with_loss = compute_loss(initial_power * self.config['NODES_PER_RECTIFIER'],
                                       self.config['RECTIFIER_LOSS_CONSTANT'],
                                       self.config['RECTIFIER_EFFICIENCY']) \
            / self.config['NODES_PER_RECTIFIER']
        return np.full(self.sc_shape, power_with_loss)

    def apply_down_nodes(self):
        """ Apply the down nodes to the power state, setting their power to zero """
        down_indices = linear_to_3d_index(self.down_nodes, self.sc_shape)
        self.power_state[down_indices] = 0
        self.rectifier_loss[down_indices] = 0
        self.sivoc_loss[down_indices] = 0

    def set_idle(self, node_indices):
        """
        Set the power consumption of specified nodes to idle.

        Parameters:
        - node_indices: list
            Indices of the nodes to set to idle.
        """
        node_indices = linear_to_3d_index(node_indices, self.sc_shape)
        self.power_state[node_indices], self.sivoc_loss[node_indices] \
            = compute_node_power(0, 0, 0, self.config)

    def update_power_state(self, scheduled_nodes, cpu_util, gpu_util, net_util):
        """
        Update the power state of scheduled nodes based on CPU and GPU utilization.
        Note: this is only used to test smart load-sharing "what-if" scenario

        Parameters:
        - scheduled_nodes: list
            Indices of the scheduled nodes.
        - cpu_util: float
            CPU utilization.
        - gpu_util: float
            GPU utilization.

        Returns:
        float
            Total power consumption of the scheduled nodes.
        """
        if len(scheduled_nodes) == 0:
            return []
        cpu_util = np.asarray(cpu_util)
        gpu_util = np.asarray(gpu_util)
        net_util = np.asarray(net_util)
        job_lengths = np.array([len(job) for job in scheduled_nodes])
        flattened_nodes = np.concatenate(scheduled_nodes, axis=0).astype(np.int64)

        cpu_util_flat = np.repeat(cpu_util, job_lengths)
        gpu_util_flat = np.repeat(gpu_util, job_lengths)
        net_util_flat = np.repeat(net_util, job_lengths)

        node_indices = linear_to_3d_index(flattened_nodes, self.config['SC_SHAPE'])
        power_value, sivoc_loss = self.power_func(cpu_util_flat, gpu_util_flat, net_util_flat, self.config)
        self.power_state[node_indices] = power_value
        self.sivoc_loss[node_indices] = sivoc_loss
        return power_value[np.cumsum(job_lengths) - 1]

    def calculate_rectifiers_needed(self, power_state_summed):
        """
        Calculate the number of rectifiers needed based on the total power consumption.

        Parameters:
        - power_state_summed: float
            Summed power consumption.

        Returns:
        int
            Number of rectifiers needed.
        """
        value = int((power_state_summed - 1) // self.config['RECTIFIER_PEAK_THRESHOLD'] + 1)
        return min(value, self.config['RECTIFIERS_PER_CHASSIS'])

    def compute_rack_power(self, smart_load_sharing=False):
        """
        Compute the power consumption of each rack in the system.

        Parameters:
        - smart_load_sharing: bool, optional
            Flag for enabling smart load sharing. Default is False.

        Returns:
        tuple
            Tuple containing rack power (kW) and rectifier losses (kW).
        """
        shape = (self.sc_shape[0], self.sc_shape[1], self.config['CHASSIS_PER_RACK'], -1)
        power_state_reshaped = np.reshape(self.power_state, shape)
        chassis_power = np.sum(power_state_reshaped, axis=-1)

        # Add in switch power
        chassis_power += self.config['SWITCHES_PER_CHASSIS'] * self.config['POWER_SWITCH']

        # Divide the power by the number of rectifiers and apply losses per rectifier
        # Smart load sharing dynamically stages rectifers as needed, e.g., when
        # all nodes are idle, only a single rectifier is used. When all
        # nodes are fully utilized, four rectifiers are used, and in between.
        if smart_load_sharing:
            vectorized_function = np.vectorize(self.calculate_rectifiers_needed)
            num_rectifiers_array = vectorized_function(chassis_power)

            # Initialize the array to hold the divided powers, using NaN for unused elements
            rectifier_power = np.full((*chassis_power.shape, self.config['RECTIFIERS_PER_CHASSIS']), np.nan)
            power_with_losses = np.copy(rectifier_power)

            # Chassis_power.shape for Frontier is (25, 3, 8)
            for i in range(chassis_power.shape[0]):
                for j in range(chassis_power.shape[1]):
                    for k in range(chassis_power.shape[2]):
                        num_rectifiers = num_rectifiers_array[i, j, k]
                        power_per_rectifier = chassis_power[i, j, k] / num_rectifiers
                        rectifier_power[i, j, k, :num_rectifiers] = power_per_rectifier
                        power_with_losses[i, j, k, :num_rectifiers] = \
                            compute_loss(power_per_rectifier,
                                         self.config['RECTIFIER_LOSS_CONSTANT'],
                                         self.config['RECTIFIER_EFFICIENCY'])

            rectifier_power = np.nan_to_num(rectifier_power)
            power_with_losses = np.nan_to_num(power_with_losses)

        else:
            divisor = np.array([4, 4, 4, 4]).reshape(1, 1, 1, 4)
            rectifier_power = chassis_power[:, :, :, np.newaxis] / divisor
            power_with_losses = compute_loss(rectifier_power,
                                             self.config['RECTIFIER_LOSS_CONSTANT'],
                                             self.config['RECTIFIER_EFFICIENCY'])

        # Compute just the losses
        rect_losses = power_with_losses - rectifier_power

        # Sum to 75 racks
        summed_power_with_losses = np.sum(power_with_losses/1000, axis=(2, 3))
        # Zero out power for missing racks
        for rack in self.config['MISSING_RACKS']:
            cdu = rack // self.config['RACKS_PER_CDU']
            rack2d = (cdu, rack % self.config['RACKS_PER_CDU'])
            summed_power_with_losses[rack2d] = 0
        summed_rect_losses = np.sum(rect_losses/1000, axis=(2, 3))

        # Add CDU numbers to table
        rows = self.sc_shape[0]
        row_numbers = np.arange(1, rows + 1).reshape(-1, 1)

        # Calculate the sum of racks 1, 2, and 3 for each row
        power_with_rows = np.hstack((row_numbers, summed_power_with_losses))
        rack_power_sum = power_with_rows[:, 1:].sum(axis=1).reshape(-1, 1)
        power_with_rows = np.hstack((power_with_rows, rack_power_sum))

        rect_loss_with_rows = np.hstack((row_numbers, summed_rect_losses))
        rack_rect_loss_sum = rect_loss_with_rows[:, 1:].sum(axis=1).reshape(-1, 1)
        rect_loss_with_rows = np.hstack((rect_loss_with_rows, rack_rect_loss_sum))

        # Return rectifier losses summed at CDU level
        return power_with_rows, rect_loss_with_rows

    def compute_sivoc_losses(self):
        """
        Compute SIVOC losses for each CDU in the system.

        Returns:
        np.ndarray
            Array containing SIVOC losses for each CDU.
        """
        # Aggregate SIVOC losses
        summed_sivoc_losses = np.sum(self.sivoc_loss / 1000, axis=2)  # kW
        rows = self.sc_shape[0]

        # Add CDU numbers to table
        row_numbers = np.arange(1, rows + 1).reshape(-1, 1)
        sivoc_loss_with_rows = np.hstack((row_numbers, summed_sivoc_losses))

        # Calculate the sum of racks 1, 2, and 3 for each row/CDU
        rack_sivoc_loss_sum = sivoc_loss_with_rows[:, 1:].sum(axis=1).reshape(-1, 1)
        sivoc_loss_with_rows = np.hstack((sivoc_loss_with_rows, rack_sivoc_loss_sum))

        return sivoc_loss_with_rows

    def get_power_df(self, rack_power, rack_loss):
        # Initialize the columns for power_df
        power_columns = self.config['POWER_DF_HEADER']
        power_data = []

        # Generate power_df
        for i, (row_pow, row_loss) in enumerate(zip(rack_power, rack_loss)):
            # Include only the required values from power row and loss row
            power_data.append((
                str(i + 1),    # CDU Number
                *row_pow[1:],  # Skip the first element of the power row (First col is CDU Number)
                *row_loss[1:]  # Skip the first element of the loss row
            ))

        power_df = pd.DataFrame(power_data, columns=power_columns)

        return power_df

    def simulate_power(self, *,
                       running_jobs,
                       scheduled_nodes,
                       cpu_utils,
                       gpu_utils,
                       net_utils
                       ):
        jobs_power = self.update_power_state(scheduled_nodes, cpu_utils, gpu_utils, net_utils)

        for i, job in enumerate(running_jobs):
            # if job.running_time % self.config['TRACE_QUANTA'] == 0:
            job.power_history.append(jobs_power[i] * len(job.scheduled_nodes))

        # Update the power array UI component
        rack_power, rect_losses = self.compute_rack_power()
        sivoc_losses = self.compute_sivoc_losses()
        rack_loss = rect_losses + sivoc_losses
        power_df = self.get_power_df(rack_power, rack_loss)

        total_power_kw = sum(row[-1] for row in rack_power) + \
            self.config['NUM_CDUS'] * self.config['POWER_CDU'] / 1000.0
        total_loss_kw = sum(row[-1] for row in rack_loss)

        # Primary return value:
        #    power_df
        # Other returns needed for further processing:
        #    rack_power,        # For cooling
        #    total_power_kw,    # For statistics
        #    total_loss_kw,     # For statistics
        #    jobs_power         # For statistics
        # ===
        return power_df, \
            rack_power, \
            total_power_kw, \
            total_loss_kw, \
            jobs_power


def record_power_stats_foreach_job(*, running_jobs, jobs_power):
    for i, job in enumerate(running_jobs):
        job.power_history.append(jobs_power[i] * len(job.scheduled_nodes))
