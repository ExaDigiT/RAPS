from enum import Enum
from ..utils import summarize_ranges

from ..workload import MAX_PRIORITY

from ..policy import PolicyType


class Scheduler:
    """ Default job scheduler with various scheduling policies. """

    def __init__(self, config, policy, resource_manager=None):
        self.config = config
        self.policy = PolicyType(policy)
        if resource_manager is None:
            raise ValueError("Scheduler requires a ResourceManager instance")
        self.resource_manager = resource_manager
        self.debug = False

    def sort_jobs(self, queue, accounts=None):
        """Sort jobs based on the selected scheduling policy."""
        if self.policy == PolicyType.FCFS or self.policy == PolicyType.BACKFILL:
            return sorted(queue, key=lambda job: job.submit_time)
        elif self.policy == PolicyType.PRIORITY:
            return sorted(queue, key=lambda job: job.priority, reverse=True)
        elif self.policy == PolicyType.FUGAKU_PTS:
            return self.sort_fugaku_redeeming(queue, accounts)
        elif self.policy == PolicyType.REPLAY:
            return sorted(queue, key=lambda job: job.start_time)
        else:
            raise ValueError(f"Policy not implemented: {self.policy}")

    def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:

            # Make sure the requested nodes are available.
            nodes_available = False
            if job.requested_nodes:  # nodes specified, i.e., telemetry replay
                if len(job.requested_nodes) <= len(self.resource_manager.available_nodes):
                    nodes_available = set(job.requested_nodes).issubset(set(self.resource_manager.available_nodes))
                else:
                    break
            else:  # synthetic
                nodes_available = len(self.resource_manager.available_nodes) >= job.nodes_required

            if self.policy == PolicyType.REPLAY and current_time < job.start_time:
                # Don't start replay jobs until they reach their start_time
                nodes_available = False

            if nodes_available:
                self.resource_manager.assign_nodes_to_job(job, current_time)
                running.append(job)
                queue.remove(job)
                if debug:
                    scheduled_nodes = summarize_ranges(job.scheduled_nodes)
                    print(f"t={current_time}: Scheduled job {job.id} with wall time {job.wall_time} on nodes {scheduled_nodes}")
            else:
                if self.policy == PolicyType.BACKFILL:
                    # Try to find a backfill candidate from the entire queue.
                    backfill_job = self.find_backfill_job(queue, len(self.resource_manager.available_nodes), current_time)
                    if backfill_job:
                        self.assign_nodes_to_job(backfill_job, self.resource_manager.available_nodes, current_time)
                        running.append(backfill_job)
                        queue.remove(backfill_job)
                        if debug:
                            scheduled_nodes = summarize_ranges(backfill_job.scheduled_nodes)
                            print(f"t={current_time}: Backfilling job {backfill_job.id} with wall time {backfill_job.wall_time} on nodes {scheduled_nodes}")


    def find_backfill_job(self, queue, num_free_nodes, current_time):
        """Finds a backfill job based on available nodes and estimated completion times.

        Based on pseudocode from Leonenkov and Zhumatiy, 'Introducing new backfill-based
        scheduler for slurm resource manager.' Procedia computer science 66 (2015): 661-669.
        """

        if not queue:
            return None

        first_job = queue[0]

        for job in queue:
            job.end_time = current_time + job.wall_time  # Estimate end time

        # Sort jobs according to their termination time (end_time)
        sorted_queue = sorted(queue, key=lambda job: job.end_time)

        # Compute shadow time by accumulating nodes
        sum_nodes = 0
        shadow_time = None
        num_extra_nodes = 0

        for job in sorted_queue:
            sum_nodes += job.nodes_required
            if sum_nodes >= first_job.nodes_required:
                shadow_time = current_time + job.wall_time
                num_extra_nodes = sum_nodes - job.nodes_required
                break

        # Find backfill job
        for job in queue:
            condition1 = job.nodes_required <= num_free_nodes and current_time + job.wall_time < shadow_time
            condition2 = job.nodes_required <= min(num_free_nodes, num_extra_nodes)

            if condition1 or condition2:
                return job

        return None

    def sort_fugaku_redeeming(self, queue, accounts=None):
        if queue == []:
            return queue
        # Priority queues not yet implemented:
        # Strategy: Sort by Fugaku Points Representing the Priority Queue
        # Everything with negative Fugaku Points get sorted according to normal priority
        priority_triple_list = []
        for job in queue:
            fugaku_priority = accounts.account_dict[job.account].fugaku_points
            # Create a tuple of the job and the priority
            priority = job.priority
            priority_triple_list.append((fugaku_priority,priority,job))
        # Sort everythin according to fugaku_points
        priority_triple_list = sorted(priority_triple_list, key=lambda x:x[0], reverse=True)
        # Find the first element with negative fugaku_points
        for cutoff, triple in enumerate(priority_triple_list):
            fugaku_priority, _, _ = triple
            if fugaku_priority < 0:
                break
        first_part = priority_triple_list[:cutoff]
        # Sort everything afterwards according to job priority
        second_part = sorted(priority_triple_list[cutoff:], key=lambda x:x[1], reverse=True)
        queue_a = []
        queue_b = []
        if first_part != []:
            _, _, queue_a = zip(*first_part)
            queue_a = list(queue_a)
        if second_part != []:
            _, _, queue_b = zip(*second_part)
            queue_b = list(queue_b)
        return queue_a + queue_b
