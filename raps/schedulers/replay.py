from ..policy import PolicyType


class Scheduler:
    """
    Mock Scheduler only considering start time.
    There is no scheduling going on but job placement according to start time.

    Default job scheduler with various scheduling policies.
    """

    def __init__(self, config, policy, resource_manager=None):
        self.config = config
        self.policy = PolicyType(policy)
        if resource_manager is None:
            raise ValueError("Scheduler requires a ResourceManager instance")
        self.resource_manager = resource_manager
        self.debug = False

    def sort_jobs(self, queue, accounts=None):
        """Sort jobs based on the selected scheduling policy."""
        return sorted(queue, key=lambda job: job.start_time)

    def prepare_system_state(self, queue, running):
        return queue

    def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        for job in queue[:]:
            # Skip jobs in queue with start time in the future
            if job.start_time >= current_time:
                continue

            nodes_available = False
            if job.nodes_required <= len(self.resource_manager.available_nodes):
                if self.policy == PolicyType.REPLAY and job.scheduled_nodes:  # Check if we need exact set
                    # is exact set available:
                    nodes_available = set(job.scheduled_nodes).issubset(set(self.resource_manager.available_nodes))
                else:
                    # we dont need the exact set:
                    nodes_available = True  # Checked above
                    if job.nodes_required == 0:
                        raise ValueError(f"Job Requested zero nodes: {job}")
                    # clear scheduled nodes
                    job.scheduled_nodes = []
            else:
                pass  # not enough nodes available

            if nodes_available:
                self.resource_manager.assign_nodes_to_job(job, current_time)
                running.append(job)
                queue.remove(job)
            else:
                # This is a replay so this should not happen
                raise ValueError(
                    f"Nodes not available!\nRequested:{job.scheduled_nodes}\n"
                    f"Available:{self.resource_manager.available_nodes}\n{job.__dict__}; "
                    f"Policy: {self.policy}")
