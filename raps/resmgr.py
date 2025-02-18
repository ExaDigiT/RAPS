import numpy as np
from .job import JobState
from scipy.stats import weibull_min


class ResourceManager:
    def __init__(self, total_nodes, down_nodes):
        self.total_nodes = total_nodes
        # Maintain a set for down nodes (e.g., nodes that are offline)
        self.down_nodes = set(down_nodes)
        # Available nodes are those that are not down
        self.available_nodes = sorted(set(range(total_nodes)) - self.down_nodes)
        # You can track system utilization history here
        self.sys_util_history = []  # list of (time, utilization) tuples

    def assign_nodes_to_job(self, job, current_time):
        """Assigns nodes to a job and updates the available nodes."""
        if len(self.available_nodes) < job.nodes_required:
            raise ValueError(f"Not enough available nodes to schedule job {job.id}")

        if job.requested_nodes:  # Telemetry replay case
            job.scheduled_nodes = job.requested_nodes
            self.available_nodes = [n for n in self.available_nodes if n not in job.scheduled_nodes]
        else:  # Synthetic or case using modified/poisson arrival times
            job.scheduled_nodes = self.available_nodes[:job.nodes_required]
            self.available_nodes = self.available_nodes[job.nodes_required:]

        # Set job start and end times according to simulation
        # This overrides actual times from telemetry and set state to 'running'
        job.start_time = current_time
        job.end_time = current_time + job.wall_time
        job.state = JobState.RUNNING  # Mark job as running

    def free_nodes_from_job(self, job):
        """Frees the nodes that were allocated to a completed job."""
        if hasattr(job, "scheduled_nodes"):
            self.available_nodes.extend(job.scheduled_nodes)
            # Remove duplicates and sort the list for consistency
            self.available_nodes = sorted(set(self.available_nodes))
        else:
            # If job has no scheduled nodes, there is nothing to free.
            pass

    def update_system_utilization(self, current_time, num_active_nodes):
        """
        Computes and records the system utilization.
        For example, utilization could be defined as the ratio of active nodes to the total non-down nodes.
        """
        # Number of nodes that are not down:
        total_operational = self.total_nodes - len(self.down_nodes)
        # Compute utilization as a percentage:
        utilization = (num_active_nodes / total_operational) * 100 if total_operational else 0
        self.sys_util_history.append((current_time, utilization))
        return utilization

    def node_failure(self, mtbf):
        """Simulate node failure using Weibull distribution."""
        shape_parameter = 1.5
        scale_parameter = mtbf * 3600  # Convert to seconds

        # Create a NumPy array of node indices, excluding down nodes
        all_nodes = np.array(sorted(set(range(self.total_nodes)) - set(self.down_nodes)))

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
            self.down_nodes.add(str(node_index))

        return newly_downed_nodes.tolist()
