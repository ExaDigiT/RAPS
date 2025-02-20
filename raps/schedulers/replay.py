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

    def prepare_system_state(self,queue,running):
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
            if job.requested_nodes:  # nodes specified, i.e., telemetry replay
                if len(job.requested_nodes) <= len(self.resource_manager.available_nodes):
                    nodes_available = set(job.requested_nodes).issubset(set(self.resource_manager.available_nodes))
                else:
                    continue   # continue instead of break, as later job with specific nodes may still be placed!
            else:  # synthetic
                if job.nodes_required:
                    pass
                else:
                    raise ValueError("No number of nodes specified.")


            if nodes_available:
                self.resource_manager.assign_nodes_to_job(job, current_time)
                running.append(job)
                queue.remove(job)
            else:
                # This is a replay so this should not happen
                raise ValueError(f"Nodes not available!\nRequested:{job.requested_nodes}\nAvailable:{self.resource_manager.available_nodes}\n{job.__dict__}")
