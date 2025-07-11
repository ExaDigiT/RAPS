import numpy as np
from .job import JobState
from scipy.stats import weibull_min


class ResourceManager:
    def __init__(self, total_nodes, down_nodes, config):
        self.total_nodes = total_nodes
        self.config = config
        self.down_nodes = set(down_nodes)
        self.nodes = []
        # Initialize nodes based on config parameters
        total_cpu_cores_per_node = self.config['CPUS_PER_NODE'] * self.config['CORES_PER_CPU']
        total_gpu_units_per_node = self.config['GPUS_PER_NODE']

        for i in range(self.total_nodes):
            is_down = i in self.down_nodes
            self.nodes.append({
                'id': i,
                'total_cpu_cores': total_cpu_cores_per_node,
                'available_cpu_cores': 0 if is_down else total_cpu_cores_per_node,
                'total_gpu_units': total_gpu_units_per_node,
                'available_gpu_units': 0 if is_down else total_gpu_units_per_node,
                'is_down': is_down
            })

        # Available nodes are now tracked by their available resources
        self.available_nodes = [node['id'] for node in self.nodes if not node['is_down']]
        self.sys_util_history = []

    def assign_nodes_to_job(self, job, current_time, node_id):
        """Assigns resources (cores, GPUs) to a job and updates the available resources."""
        # For multitenancy, a job is assigned to a single node.
        # We need to find a node that can satisfy the job's resource requirements.

        found_node = None
        # Use the provided node_id directly
        if node_id is not None and node_id < len(self.nodes) and not self.nodes[node_id]['is_down']:
            node = self.nodes[node_id]
            if (node['available_cpu_cores'] >= job.cpu_cores_required and
                    node['available_gpu_units'] >= job.gpu_units_required):
                found_node = node

        if found_node is None:
            raise ValueError(f"Not enough available resources to schedule job {job.id} on node {node_id}.")

        # Allocate resources on the found node
        found_node['available_cpu_cores'] -= job.cpu_cores_required
        found_node['available_gpu_units'] -= job.gpu_units_required

        # Assign the node and allocated resources to the job
        job.scheduled_nodes = [found_node['id']]
        job.allocated_cpu_cores = job.cpu_cores_required
        job.allocated_gpu_units = job.gpu_units_required

        # Set job start and end times according to simulation
        job.start_time = current_time
        job.end_time = current_time + job.wall_time
        job.state = JobState.RUNNING  # Mark job as running

    def free_nodes_from_job(self, job):
        """Frees the resources (cores, GPUs) that were allocated to a completed job."""
        if hasattr(job, "scheduled_nodes") and job.scheduled_nodes:
            node_id = job.scheduled_nodes[0] # Assuming a job is scheduled on a single node
            if node_id < len(self.nodes):
                node = self.nodes[node_id]
                node['available_cpu_cores'] += job.allocated_cpu_cores
                node['available_gpu_units'] += job.allocated_gpu_units
            else:
                print(f"Warning: Job {job.id} scheduled on non-existent node {node_id}. Cannot free resources.")
        else:
            # If job has no scheduled nodes, there is nothing to free.
            pass

    def update_system_utilization(self, current_time, running_jobs):
        """
        Computes and records the system utilization based on allocated CPU cores and GPU units.
        """
        total_cpu_cores = sum(node['total_cpu_cores'] for node in self.nodes)
        total_gpu_units = sum(node['total_gpu_units'] for node in self.nodes)

        allocated_cpu_cores = sum(job.allocated_cpu_cores for job in running_jobs)
        allocated_gpu_units = sum(job.allocated_gpu_units for job in running_jobs)

        cpu_utilization = (allocated_cpu_cores / total_cpu_cores) * 100 if total_cpu_cores else 0
        gpu_utilization = (allocated_gpu_units / total_gpu_units) * 100 if total_gpu_units else 0

        # For now, we'll just use CPU utilization as the primary system utilization metric
        # You might want to combine these or choose a different primary metric
        self.sys_util_history.append((current_time, cpu_utilization))
        return cpu_utilization

    def node_failure(self, mtbf):
        """Simulate node failure using Weibull distribution."""
        shape_parameter = 1.5
        scale_parameter = mtbf * 3600  # Convert to seconds

        # Create a NumPy array of node indices, excluding already down nodes
        operational_node_ids = np.array([node['id'] for node in self.nodes if not node['is_down']])

        if len(operational_node_ids) == 0:
            return [] # No operational nodes to fail

        # Sample the Weibull distribution for all operational nodes at once
        random_values = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=len(operational_node_ids))

        # Identify nodes that have failed (using a threshold for demonstration)
        failure_threshold = 0.001 # This threshold might need tuning
        failed_nodes_mask = random_values < failure_threshold
        newly_downed_node_ids = operational_node_ids[failed_nodes_mask]

        # Update the state of the newly downed nodes in self.nodes
        for node_id in newly_downed_node_ids:
            node = self.nodes[node_id]
            node['is_down'] = True
            node['available_cpu_cores'] = 0
            node['available_gpu_units'] = 0
            self.down_nodes.add(node_id) # Add to the set of down node IDs

        return newly_downed_node_ids.tolist()
