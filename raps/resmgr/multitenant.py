import numpy as np
from ..job import JobState
from scipy.stats import weibull_min


def assert_node_accounting_ok(node):
    assert node['available_cpu_cores'] >= 0, "available_cpu_cores went negative"
    assert node['available_gpu_units'] >= 0, "available_gpu_units went negative"


class MultiTenantResourceManager:
    """
    Resource manager for per-node CPU/GPU multitenancy.
    """

    def __init__(self, total_nodes, down_nodes, config):
        self.total_nodes = total_nodes
        self.config = config
        self.down_nodes = set(down_nodes)
        self.nodes = []
        # Track total allocations for reporting
        self.allocated_cpu_cores = 0
        self.allocated_gpu_units = 0
        self.sys_util_history = []

        # Determine per-node capacities
        total_cpu = self.config['CPUS_PER_NODE'] * self.config['CORES_PER_CPU']
        total_gpu = self.config.get('GPUS_PER_NODE', 0)

        # Initialize node state
        for i in range(self.total_nodes):
            is_down = i in self.down_nodes
            self.nodes.append({
                'id': i,
                'total_cpu_cores':     total_cpu,
                'available_cpu_cores': 0 if is_down else total_cpu,
                'total_gpu_units':     total_gpu,
                'available_gpu_units': 0 if is_down else total_gpu,
                'is_down':             is_down
            })

        # List of up nodes for quick enumeration
        self.available_nodes = [n['id'] for n in self.nodes if not n['is_down']]

    def assign_nodes_to_job(self, job, current_time, node_id=None):
        """Assigns cores/GPUs to a job on one eligible node."""
        # Try preferred node
        found = None
        if node_id is not None and 0 <= node_id < len(self.nodes):
            candidate = self.nodes[node_id]
            if (not candidate['is_down'] and
                candidate['available_cpu_cores'] >= job.cpu_cores_required and
                    candidate['available_gpu_units'] >= job.gpu_units_required):
                found = candidate

        # Fallback: first-fit
        if found is None:
            for candidate in self.nodes:
                if (not candidate['is_down'] and
                    candidate['available_cpu_cores'] >= job.cpu_cores_required and
                        candidate['available_gpu_units'] >= job.gpu_units_required):
                    found = candidate
                    break

        if found is None:
            raise ValueError(f"Not enough available resources to schedule job {job.id}.")

        # Allocate resources
        found['available_cpu_cores'] -= job.cpu_cores_required
        found['available_gpu_units'] -= job.gpu_units_required
        self.allocated_cpu_cores += job.cpu_cores_required
        self.allocated_gpu_units += job.gpu_units_required

        # ---- Invariant checks (after mutating node/RM state) ----
        assert_node_accounting_ok(found)  # no negatives left
        assert self.allocated_cpu_cores >= 0 and self.allocated_gpu_units >= 0
        # Optional: global sanity vs. totals
        assert self.allocated_cpu_cores <= sum(n['total_cpu_cores'] for n in self.nodes)
        assert self.allocated_gpu_units <= sum(n['total_gpu_units'] for n in self.nodes)

        # Record on job
        job.scheduled_nodes = [found['id']]
        job.allocated_cpu_cores = job.cpu_cores_required
        job.allocated_gpu_units = job.gpu_units_required
        job.start_time = current_time
        if job.expected_run_time:
            job.end_time = current_time + job.expected_run_time  # this may be an assumption (See default.py)
        job.current_state = JobState.RUNNING

    def free_nodes_from_job(self, job):
        """Releases cores/GPUs from a completed job."""
        if getattr(job, 'scheduled_nodes', None):
            nid = job.scheduled_nodes[0]
            if 0 <= nid < len(self.nodes):
                node = self.nodes[nid]
                node['available_cpu_cores'] += getattr(job, 'allocated_cpu_cores', 0)
                node['available_gpu_units'] += getattr(job, 'allocated_gpu_units', 0)
                self.allocated_cpu_cores -= getattr(job, 'allocated_cpu_cores', 0)
                self.allocated_gpu_units -= getattr(job, 'allocated_gpu_units', 0)
            else:
                print(f"Warning: Job {job.id} had invalid node {nid} during free.")

    def update_system_utilization(self, current_time, running_jobs):
        """
        Computes and records utilization based on allocated CPU/GPU across all nodes.
        """
        total_cpu = sum(n['total_cpu_cores'] for n in self.nodes)
        total_gpu = sum(n['total_gpu_units'] for n in self.nodes)
        used_cpu = self.allocated_cpu_cores
        used_gpu = self.allocated_gpu_units

        cpu_util = (used_cpu / total_cpu) * 100 if total_cpu else 0
        gpu_util = (used_gpu / total_gpu) * 100 if total_gpu else 0

        # Choose GPU util if GPUs exist, else CPU
        util = gpu_util if self.config.get('GPUS_PER_NODE', 0) > 0 else cpu_util
        self.sys_util_history.append((current_time, util))
        return util

    def node_failure(self, mtbf):
        """
        Simulate random node failures via a Weibull distribution.
        """
        shape = 1.5
        scale = mtbf * 3600
        ops = np.array([n['id'] for n in self.nodes if not n['is_down']])
        if ops.size == 0:
            return []

        vals = weibull_min.rvs(shape, scale=scale, size=ops.size)
        failed = ops[vals < 0.001]
        for nid in failed:
            node = self.nodes[nid]
            node['is_down'] = True
            node['available_cpu_cores'] = 0
            node['available_gpu_units'] = 0
            self.down_nodes.add(nid)
        return failed.tolist()
