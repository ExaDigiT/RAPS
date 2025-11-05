from typing import List
from ..utils import summarize_ranges
from ..policy import PolicyType, BackfillType


class Scheduler:
    """ Default job scheduler with various scheduling policies. """

    def __init__(self, config, policy, bfpolicy=None, jobs=None, resource_manager=None):
        self.config = config
        if policy is None:  # policy is passed as policy=None, therefore default is not choosen
            policy = "replay"
        self.policy = PolicyType(policy)
        self.bfpolicy = BackfillType(bfpolicy)
        if resource_manager is None:
            raise ValueError("Scheduler requires a ResourceManager instance")
        self.resource_manager = resource_manager
        self.debug = False

    def sort_jobs(self, queue, accounts=None):
        """Sort jobs based on the selected scheduling policy."""
        if self.policy == PolicyType.FCFS:
            return sorted(queue, key=lambda job: job.submit_time)
        elif self.policy == PolicyType.PRIORITY:
            return sorted(queue, key=lambda job: job.priority, reverse=True)
        elif self.policy == PolicyType.SJF:
            return sorted(queue, key=lambda job: job.time_limit)
        elif self.policy == PolicyType.LJF:
            return sorted(queue, key=lambda job: job.nodes_required, reverse=True)
        elif self.policy == PolicyType.REPLAY:
            return sorted(queue, key=lambda job: job.start_time)
        else:
            raise ValueError(f"Policy not implemented: {self.policy}")

    def schedule(self, queue, running, current_time, accounts=None, sorted=False):
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:
            if self.debug:
                print(
                    f"[DEBUG] Scheduler: Considering job {job.id} "
                    f"(CPU: {job.cpu_cores_required}, GPU: {job.gpu_units_required})")
            if self.policy == PolicyType.REPLAY:
                if job.start_time > current_time:
                    continue  # Replay: Job didn't start yet. Next!
                else:
                    pass
            else:
                pass

            nodes_available = self.check_available_nodes(job)

            if nodes_available is not None:
                self.place_job_and_manage_queues(job, queue, running, current_time, nodes_available)
            else:  # In case the job was not placed, see how we should continue:
                if self.bfpolicy is not None:
                    backfill_job, node_id = self.backfill(queue, running, current_time)
                    if backfill_job and node_id is not None:
                        self.place_job_and_manage_queues(backfill_job, queue, running, current_time, node_id)

                # After backfill dedice continue processing the queue or wait, continuing may result in fairness issues.
                if self.policy in [PolicyType.REPLAY]:
                    # print(f"Nodes available {nodes_available} - "
                    #       f"Req:{len(job.requested_nodes)} N-avail:{len(self.resource_manager.available_nodes)}")
                    continue  # Regardless if the job at the front of the queue doenst fit, try placing all of them.
                elif self.policy in [PolicyType.FCFS, PolicyType.PRIORITY,
                                     PolicyType.LJF, PolicyType.SJF]:
                    break  # The job at the front of the queue doesnt fit stop processing the queue.
                else:
                    raise NotImplementedError(
                        "Depending on the Policy this choice should be explicit. Add the implementation above!")

    def prepare_system_state(self, jobs_to_submit: List, running, timestep_start):
        # def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
        """
        In the case of replay and fast forward, previously placed jobs should be present.

        """
        if self.policy == PolicyType.REPLAY:
            total_jobs = len(jobs_to_submit)
            print(f"All jobs: {total_jobs}")

            # Keep only jobs have an end time in the future future.
            jobs_to_submit[:] = [job for job in jobs_to_submit if job['end_time'] >= timestep_start]
            print(f"Num jobs in the past: {total_jobs - len(jobs_to_submit)}")

            # Identify jobs that started in the past and Split them from the jobs that will start in the future:
            jobs_to_start_now = [job for job in jobs_to_submit if job['start_time'] < timestep_start]
            print(f"Num jobs that started in the past: {len(jobs_to_start_now)}")

            jobs_to_submit[:] = [job for job in jobs_to_submit if job['start_time'] >= timestep_start]
            print(f"Num jobs to be schedule in the simulation: {len(jobs_to_submit)}")

            # Now schedule them with their orignal start time.
            # This has to be done one by one!
            for job in jobs_to_start_now:
                self.schedule([job], running, job['start_time'], sorted=True)
            # self.schedule(jobs_to_start_now, running, 0, False)
            return jobs_to_submit
        else:
            return jobs_to_submit

    def place_job_and_manage_queues(self, job, queue, running, current_time, node_id):
        self.resource_manager.assign_nodes_to_job(job, current_time, node_id)
        running.append(job)
        queue.remove(job)
        if self.debug:
            scheduled_nodes = summarize_ranges(job.scheduled_nodes)
            print(f"t={current_time}: Scheduled job {job.id} with wall time {job.wall_time} on nodes {scheduled_nodes}")

    def check_available_nodes(self, job):
        """Checks if there are available resources (CPU cores, GPU units) for the job on any node."""
        # Iterate through all nodes managed by the ResourceManager
        for node in self.resource_manager.nodes:
            if self.debug:
                print(
                    f"[DEBUG]   Checking node {node['id']}: "
                    f"Available CPU: {node['available_cpu_cores']}, "
                    f"Available GPU: {node['available_gpu_units']}. "
                    f"Job needs CPU: {job.cpu_cores_required}, GPU: {job.gpu_units_required}")
            # Skip if the node is down
            if node['is_down']:
                continue

            # Check if the node has enough available CPU cores and GPU units
            if (node['available_cpu_cores'] >= job.cpu_cores_required and
                    node['available_gpu_units'] >= job.gpu_units_required):
                # If a suitable node is found, return its ID
                return node['id']
        # If no suitable node is found, return None
        return None

    def backfill(self, queue: List, running: List, current_time):
        # Try to find a backfill candidate from the entire queue.
        while queue:
            backfill_job, node_id = self.find_backfill_job(queue, running, current_time)
            if backfill_job is not None and node_id is not None:
                # Instead of placing here, return the job and node_id to the caller
                return backfill_job, node_id
            else:
                break
        return None, None

    def find_backfill_job(self, queue, running, current_time):
        """Finds a backfill job based on available nodes and estimated completion times.

        Loosely based on pseudocode from Leonenkov and Zhumatiy, 'Introducing new backfill-based
        scheduler for slurm resource manager.' Procedia computer science 66 (2015): 661-669.
        """
        if not queue:
            return None, None

        # Identify when the nex job in the queue could run as a time limit:
        # first_job = queue[0]  # Unused
        # For multitenancy, we need to check if the first job can fit on any node
        # based on its core/GPU requirements, not just nodes_required.
        # This is a simplification; a more complex backfill might consider
        # if the job can fit by combining resources from multiple nodes.
        # For now, we assume it needs to fit on a single node.

        # We need to know the total available resources if all running jobs finish by shadow_time_end
        # This is complex with multitenancy, so for now, we'll simplify the backfill logic
        # to just check if a job can fit on *any* node, not necessarily the one
        # that will be freed up by the first job in line.

        # The original logic for shadow_time_end and shadow_nodes_avail is based on whole nodes.
        # With multitenancy, this needs a more sophisticated resource projection.
        # For now, we will make `time_limit` effectively infinite for backfill candidates
        # if the job can fit on *any* node, and rely on `check_available_nodes`.

        # Revert to a simpler time_limit for now, or remove it if not applicable
        # For now, let's assume time_limit is not strictly tied to node availability
        # in the same way as before, and focus on resource availability.
        time_limit = float('inf')  # Effectively no time limit for backfill candidates

        # We now have the time_limit after which no backfilled job should end
        # as the next job in line has the necessary resrouces after this time limit.

        # Find and return the first job that fits
        if self.bfpolicy == BackfillType.NONE:
            pass
        elif self.bfpolicy == BackfillType.EASY:
            queue[:] = sorted(queue, key=lambda job: job.submit_time)
            return self.return_first_fit(queue, time_limit)
        elif self.bfpolicy == BackfillType.FIRSTFIT:
            pass  # Stay with the prioritization!
            return self.return_first_fit(queue, time_limit)
        elif self.bfpolicy in [BackfillType.BESTFIT,
                               BackfillType.GREEDY,
                               BackfillType.CONSERVATIVE,
                               ]:
            raise NotImplementedError(f"{self.bfpolicy} not implemented! Please implement!")
        else:
            raise NotImplementedError(f"{self.bfpolicy} not implemented.")
        return None, None

    def return_first_fit(self, queue, time_limit):
        for job in queue:
            # Check if the job can fit on any node based on its resource requirements
            node_id = self.check_available_nodes(job)
            if node_id is not None:
                # If a suitable node is found, return the job and the node_id
                return job, node_id
        return None, None
