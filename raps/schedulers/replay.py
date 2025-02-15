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

### NOTE:
# Both schdule and schedule_v2 do not work, as the resource_manager claims nodes not available.
# This needs to be fixed.

    def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
    #### DOES NOT WORK, Nodes are not available! in resrouce_manager.assign_nodes_to_job!
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        # Filter Jobs with start_time in this epoch
        queue[:] = [job for job in queue if job.start_time <= current_time]

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:
            nodes_available = set(job.requested_nodes).issubset(set(self.resource_manager.available_nodes))
            self.resource_manager.assign_nodes_to_job(job, current_time)
            running.append(job)
            queue.remove(job)
            continue

    def schedule_v2(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
    #### DOES NOT WORK, Nodes are not available!
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        # Filter Jobs with start_time in this epoch
        queue[:] = [job for job in queue if job.start_time <= current_time]

        for job in queue[:]:
            nodes_available = False
            if job.requested_nodes:  # nodes specified, i.e., telemetry replay
                if len(job.requested_nodes) <= len(self.resource_manager.available_nodes):
                    nodes_available = set(job.requested_nodes).issubset(set(self.resource_manager.available_nodes))
                else:
                    continue   # continue instead of break, as later job with specific nodes may still be placed!
            else:  # synthetic
                raise ValueError("No jobs requested?")

            if nodes_available:
                self.resource_manager.assign_nodes_to_job(job, current_time)
                running.append(job)
                queue.remove(job)
            else:
                raise ValueError("Nodes not available!")  # Jobs may be queued
                pass  # Try next time
