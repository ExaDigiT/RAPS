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
            # job.start_time is None when telemetry.update_jobs() has
            # replaced the recorded arrival process with synthetic
            # (--arrival poisson) arrivals: it deliberately clears
            # start_time/end_time since the original observed start is no
            # longer meaningful, and only submit_time drives arrival order
            # in that mode. Fall back to submit_time so replay ordering
            # still has a well-defined key instead of crashing (or, pre-fix,
            # crashing one level up in schedule() on `job.start_time >
            # current_time` for the same reason).
            return sorted(
                queue,
                key=lambda job: job.start_time if job.start_time is not None else job.submit_time,
            )
        else:
            raise ValueError(f"Policy not implemented: {self.policy}")

    def schedule(self, queue, running, current_time, accounts=None, sorted=False):
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue, accounts)

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:
            if self.policy == PolicyType.REPLAY:
                # See sort_jobs() above: start_time is None under synthetic
                # (--arrival poisson) arrivals, where there is no recorded
                # start to gate on -- treat that as "eligible now" rather
                # than crashing on None > current_time.
                if job.start_time is not None and job.start_time > current_time:
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
                if self.bfpolicy != BackfillType.NONE:  # bfpolicy is always a BackfillType, never None
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
        """
        A job that can start now without delaying the job at the head of the queue (EASY
        backfill: Lifka 1995; Leonenkov and Zhumatiy, Procedia Computer Science 66 (2015) 661-669).

        The head job is promised the shadow time: when enough running jobs have reached their
        time limits for it to fit. A candidate may start now if it fits in the free nodes and
        either ends (by its time limit) before the shadow time, or needs no more than the extra
        nodes the head job will leave unused at the shadow time. FIRSTFIT tries candidates in
        queue (priority) order, EASY in submit order; the queue itself is not reordered, so the
        head job keeps its place on the next call.
        """
        if not queue or self.bfpolicy == BackfillType.NONE:
            return None
        if self.bfpolicy in [BackfillType.BESTFIT, BackfillType.GREEDY, BackfillType.CONSERVATIVE]:
            raise NotImplementedError(f"{self.bfpolicy} not implemented! Please implement!")
        if self.bfpolicy not in [BackfillType.EASY, BackfillType.FIRSTFIT]:
            raise NotImplementedError(f"{self.bfpolicy} not implemented.")

        first_job = queue[0]
        replay_nodes = self.policy == PolicyType.REPLAY and first_job.scheduled_nodes
        nodes_required = len(first_job.scheduled_nodes) if replay_nodes else first_job.nodes_required

        # Running jobs in the order they reach their time limits (what a live scheduler knows)
        shadow_time = float("inf")  # never enough nodes for the head job: nothing to protect
        shadow_nodes = len(self.resource_manager.available_nodes)
        for job in sorted(running, key=lambda job: job.start_time + job.time_limit):
            shadow_nodes += job.nodes_required
            if shadow_nodes >= nodes_required:
                shadow_time = job.start_time + job.time_limit
                break
        # Nodes free at the shadow time beyond what the head job needs. A replayed head job needs
        # specific nodes, so counting spare nodes says nothing there.
        extra_nodes = 0 if replay_nodes else shadow_nodes - nodes_required
        if shadow_time == float("inf"):
            extra_nodes = float("inf")

        candidates = queue[1:]
        if self.bfpolicy == BackfillType.EASY:
            candidates = sorted(candidates, key=lambda job: job.submit_time)
        return self.return_first_fit(candidates, shadow_time - current_time, extra_nodes)

    def return_first_fit(self, queue, time_limit, extra_nodes=0):
        """First job that fits now and ends within time_limit or needs at most extra_nodes."""
        for job in queue:
            if job.time_limit <= time_limit or job.nodes_required <= extra_nodes:
                if self.check_available_nodes(job):
                    return job
        return None
