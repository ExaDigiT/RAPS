import numpy as np
from .utils import linear_to_3d_index


class FLOPSManager():

    def __init__(self, **kwargs):
        self.config = kwargs.get('config')
        self.validate = kwargs.get('validate')
        self.flop_state = np.zeros(self.config['SC_SHAPE'])

    def update_flop_state(self, scheduled_nodes, cpu_util, gpu_util):
        if len(scheduled_nodes) == 0:
            return
        cpu_util = np.asarray(cpu_util)
        gpu_util = np.asarray(gpu_util)
        job_lengths = np.array([len(job) for job in scheduled_nodes])
        flattened_nodes = np.concatenate(scheduled_nodes, axis=0).astype(np.int64)

        cpu_util_flat = np.repeat(cpu_util, job_lengths)
        gpu_util_flat = np.repeat(gpu_util, job_lengths)

        node_indices = linear_to_3d_index(flattened_nodes, self.config['SC_SHAPE'])

        if self.validate:   # cpu_util is in fact node_Watts in this case
            total_peak = (
                self.config['CPU_FP_RATIO'] * self.config['CPU_PEAK_FLOPS'] +
                self.config['GPU_FP_RATIO'] * self.config['GPU_PEAK_FLOPS']
            )
            denominator = (
                self.config['POWER_CPU_MAX'] * self.config['CPUS_PER_NODE'] +
                self.config['POWER_GPU_MAX'] * self.config['GPUS_PER_NODE'] +
                self.config['POWER_NIC'] * self.config['NICS_PER_NODE'] +
                self.config['POWER_NVME']
            )
            self.flop_state[node_indices] = total_peak * (cpu_util_flat / denominator)
        else:
            self.flop_state[node_indices] = (
                self.config['CPU_FP_RATIO'] * cpu_util_flat * self.config['CPU_PEAK_FLOPS'] +
                self.config['GPU_FP_RATIO'] * gpu_util_flat * self.config['GPU_PEAK_FLOPS']
            )

    def get_rpeak(self):
        node_peak_flops = (
            self.config['CPUS_PER_NODE'] * self.config['CPU_PEAK_FLOPS'] +
            self.config['GPUS_PER_NODE'] * self.config['GPU_PEAK_FLOPS']
        )
        system_peak_flops = self.config['AVAILABLE_NODES'] * node_peak_flops
        return system_peak_flops

    def get_system_performance(self):
        return np.sum(self.flop_state)

    def simulate_flops(self, *, scheduled_nodes, cpu_util, gpu_util, total_power_kw):
        self.update_flop_state(scheduled_nodes=scheduled_nodes,
                               cpu_util=cpu_util,
                               gpu_util=gpu_util)
        pflops = self.get_system_performance() / 1E15
        if total_power_kw != 0:
            gflops_per_watt = pflops * 1E6 / (total_power_kw * 1000)
        else:
            gflops_per_watt = 0
        return pflops, gflops_per_watt
