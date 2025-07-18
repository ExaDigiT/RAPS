import numpy as np
from .job import JobState
from scipy.stats import weibull_min


class ResourceManager:
    def __init__(self, total_nodes, down_nodes, config):
        self.total_nodes         = total_nodes
        self.config              = config
        self.multitenant         = bool(self.config.get("multitenant", False))
        self.down_nodes          = set(down_nodes)

        # Track allocated resources for querying
        self.allocated_cpu_cores = 0
        self.allocated_gpu_units = 0
        self.sys_util_history    = []

        # Compute per-node capacities: support multiple config styles
        if 'CORES_PER_CPU' in config and 'CPUS_PER_NODE' in config:
            # CPUS_PER_NODE = # sockets, CORES_PER_CPU = cores per socket
            total_cpu = config['CPUS_PER_NODE'] * config['CORES_PER_CPU']
        else:
            # Either flat CORES_PER_NODE or CPUS_PER_NODE used as total cores
            total_cpu = config.get('CORES_PER_NODE', config.get('CPUS_PER_NODE', 0))
        total_gpu = config.get('GPUS_PER_NODE', 0)

        # Build a unified node list (always present) so engine can inspect it
        self.nodes = []
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

        # Legacy whole-node allocation tracking
        if not self.multitenant:
            self.available_nodes = [node['id'] for node in self.nodes if not node['is_down']]

    def assign_nodes_to_job(self, job, current_time, node_id=None):
        if not self.multitenant:
            return self._assign_whole_node(job, current_time)

        # Multitenant allocation path
        found_node = None
        # Try specific node_id if provided
        if node_id is not None and 0 <= node_id < len(self.nodes):
            node = self.nodes[node_id]
            if (not node['is_down'] and
                node['available_cpu_cores'] >= job.cpu_cores_required and
                node['available_gpu_units']  >= job.gpu_units_required):
                found_node = node

        # Fallback: scan all nodes
        if found_node is None:
            for node in self.nodes:
                if (not node['is_down'] and
                    node['available_cpu_cores'] >= job.cpu_cores_required and
                    node['available_gpu_units']  >= job.gpu_units_required):
                    found_node = node
                    break

        if found_node is None:
            raise ValueError(f"Not enough available resources to schedule job {job.id}.")

        # Allocate resources
        found_node['available_cpu_cores'] -= job.cpu_cores_required
        found_node['available_gpu_units'] -= job.gpu_units_required
        job.scheduled_nodes       = [found_node['id']]
        job.allocated_cpu_cores   = job.cpu_cores_required
        job.allocated_gpu_units   = job.gpu_units_required
        self.allocated_cpu_cores += job.cpu_cores_required
        self.allocated_gpu_units += job.gpu_units_required

        job.start_time = current_time
        job.end_time   = current_time + job.wall_time
        job.state      = JobState.RUNNING

    def _assign_whole_node(self, job, current_time):
        # Legacy whole-node allocation supporting explicit list or count-based mode
        # 1) If replaying specific nodes, use requested_nodes
        if getattr(job, 'requested_nodes', None):
            take  = len(job.requested_nodes)
            picks = job.requested_nodes
        # 2) If the job carries a nodes_alloc attribute, honor it
        elif hasattr(job, 'nodes_alloc'):
            take  = job.nodes_alloc
            picks = self.available_nodes[:take]
        # 3) Otherwise fall back to nodes_required
        else:
            take  = job.nodes_required
            picks = self.available_nodes[:take]

        # Ensure we have enough free nodes
        if take > len(self.available_nodes):
            raise ValueError(f"Not enough available nodes to schedule job {job.id}: "
                             f"needs {take}, only {len(self.available_nodes)} free")

        # Allocate
        job.scheduled_nodes    = picks
        self.available_nodes   = [n for n in self.available_nodes if n not in picks]
        job.start_time         = current_time
        job.end_time           = current_time + job.wall_time
        job.state              = JobState.RUNNING

    def free_nodes_from_job(self, job):
        """Frees the resources (whole-node or multitenant) allocated to a completed job."""
        if not self.multitenant:
            # DEBUG: show what we're freeing
            print(f"[DEBUG free] Job {job.id} releasing nodes: {getattr(job, 'scheduled_nodes', None)}")
            print(f"[DEBUG free] Available before release: {self.available_nodes}")
            self._free_whole_nodes(job)
            print(f"[DEBUG free] Available after release: {self.available_nodes}")
            return

        # Multitenant release path
        if hasattr(job, "scheduled_nodes") and job.scheduled_nodes:
            node_id = job.scheduled_nodes[0]
            print(f"[DEBUG free] Job {job.id} releasing multitenant node: {node_id}")
            node = self.nodes[node_id] if 0 <= node_id < len(self.nodes) else None
            if node:
                before_cpu = node['available_cpu_cores']
                before_gpu = node['available_gpu_units']
                node['available_cpu_cores'] += getattr(job, 'allocated_cpu_cores', 0)
                node['available_gpu_units']  += getattr(job, 'allocated_gpu_units', 0)
                self.allocated_cpu_cores    -= getattr(job, 'allocated_cpu_cores', 0)
                self.allocated_gpu_units    -= getattr(job, 'allocated_gpu_units', 0)
                print(f"[DEBUG free] Node {node_id} before (cpu,gpu)=({before_cpu},{before_gpu}), after=({node['available_cpu_cores']},{node['available_gpu_units']})")
            else:
                print(f"Warning: Job {job.id} scheduled on invalid node {node_id}")

    def _free_whole_nodes(self, job):
        # Legacy free whole nodes
        if hasattr(job, "scheduled_nodes"):
            for n in job.scheduled_nodes:
                if n not in self.available_nodes:
                    self.available_nodes.append(n)
            self.available_nodes = sorted(self.available_nodes)

    def update_system_utilization(self, current_time, running_jobs):
        """
        Computes and records the system utilization.
        If running in whole-node mode, uses node-based utilization; otherwise uses core/GPU utilization.
        """
        if not self.multitenant:
            # Whole-node utilization: percentage of active nodes
            num_active = len(running_jobs)
            return self._update_whole_node_util(current_time, num_active)

        # Multitenant utilization: based on CPU/GPU usage
        total_cpu     = sum(n['total_cpu_cores'] for n in self.nodes)
        total_gpu     = sum(n['total_gpu_units']      for n in self.nodes)
        allocated_cpu = self.allocated_cpu_cores
        allocated_gpu = self.allocated_gpu_units
        cpu_util      = (allocated_cpu / total_cpu) * 100 if total_cpu else 0
        gpu_util      = (allocated_gpu / total_gpu) * 100 if total_gpu else 0
        # Choose GPU utilization if GPUs are present
        util = gpu_util if self.config.get('GPUS_PER_NODE', 0) > 0 else cpu_util
        self.sys_util_history.append((current_time, util))
        return util

    def _update_whole_node_util(self, current_time, num_active_nodes):
        operational = self.total_nodes - len(self.down_nodes)
        util        = (num_active_nodes / operational) * 100 if operational else 0
        self.sys_util_history.append((current_time, util))
        return util

    def node_failure(self, mtbf):
        if not self.multitenant:
            # Legacy node failure sampling on whole nodes
            available = np.array([n for n in range(self.total_nodes) if n not in self.down_nodes])
            if available.size == 0:
                return []
            shape_param = 1.5
            scale_param = mtbf * 3600
            random_vals = weibull_min.rvs(shape_param, scale=scale_param, size=available.size)
            failure_threshold = 0.001
            failed = available[random_vals < failure_threshold]
            for nid in failed:
                if nid in self.available_nodes:
                    self.available_nodes.remove(nid)
                self.down_nodes.add(nid)
            return failed.tolist()

        # Multitenant node failure sampling
        operational_ids = np.array([n['id'] for n in self.nodes if not n['is_down']])
        if operational_ids.size == 0:
            return []
        shape_param = 1.5
        scale_param = mtbf * 3600
        random_vals = weibull_min.rvs(shape_param, scale=scale_param, size=operational_ids.size)
        failure_threshold = 0.001
        failed = operational_ids[random_vals < failure_threshold]
        for node_id in failed:
            node = self.nodes[node_id]
            node['is_down']            = True
            node['available_cpu_cores'] = 0
            node['available_gpu_units'] = 0
            self.down_nodes.add(node_id)
        return failed.tolist()
