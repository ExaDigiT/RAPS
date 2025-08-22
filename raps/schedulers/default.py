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
            if self.policy == PolicyType.REPLAY:
                if job.start_time > current_time:
                    continue  # Replay: Job didn't start yet. Next!
                else:
                    # assert job.start_time == current_time, f"{job.start_time} == {current_time}"
                    pass
            else:
                pass

            nodes_available = self.check_available_nodes(job)

            if nodes_available:
                self.place_job_and_manage_queues(job, queue, running, current_time)
            else:  # In case the job was not placed, see how we should continue:
                if self.bfpolicy is not None or self.bfpolicy is not BackfillType.NONE:
                    self.backfill(queue, running, current_time)

                # After backfill dedice continue processing the queue or wait, continuing may result in fairness issues.
                if self.policy in [PolicyType.REPLAY]:
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

    def place_job_and_manage_queues(self, job, queue, running, current_time):
        self.resource_manager.assign_nodes_to_job(job, current_time, self.policy)
        running.append(job)
        queue.remove(job)
        if self.debug:
            scheduled_nodes = summarize_ranges(job.scheduled_nodes)
            print(f"t={current_time}: Scheduled job {job.id} with time limit "
                  f"{job.time_limit} on nodes {scheduled_nodes}")

    def check_available_nodes(self, job):
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
        return nodes_available

    def backfill(self, queue: List, running: List, current_time):
        # Try to find a backfill candidate from the entire queue.
        while queue:
            backfill_job = self.find_backfill_job(queue, running, current_time)
            if backfill_job:
                self.place_job_and_manage_queues(backfill_job, queue, running, current_time)
            else:
                break

    def find_backfill_job(self, queue, running, current_time):
        """Finds a backfill job based on available nodes and estimated completion times.

        Loosely based on pseudocode from Leonenkov and Zhumatiy, 'Introducing new backfill-based
        scheduler for slurm resource manager.' Procedia computer science 66 (2015): 661-669.
        """
        if not queue:
            return None

        # Identify when the nex job in the queue could run as a time limit:
        first_job = queue[0]
        nodes_required = 0
        if self.policy == PolicyType.REPLAY and first_job.scheduled_nodes:  # This needs to be done propper!
            nodes_required = len(first_job.scheduled_nodes)
        else:
            nodes_required = first_job.nodes_required

        sorted_running = sorted(running, key=lambda job: job.time_limit)

        # Identify when we have enough nodes therefore the start time of the first_job in line
        shadow_time_end = 0
        shadow_nodes_avail = len(self.resource_manager.available_nodes)
        for job in sorted_running:
            if shadow_nodes_avail >= nodes_required:
                break
            else:
                shadow_nodes_avail += job.nodes_required
                shadow_time_end = job.start_time + job.time_limit

        time_limit = shadow_time_end - current_time
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

    def return_first_fit(self, queue, time_limit):
        for job in queue:
            if job.time_limit <= time_limit:
                nodes_available = self.check_available_nodes(job)
                if nodes_available:
                    return job
                else:
                    continue
            else:
                continue
        return None
