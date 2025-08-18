from third_party.ScheduleFlow import ScheduleFlow
from third_party.ScheduleFlow import _intScheduleFlow
from third_party.ScheduleFlow._intScheduleFlow import EventType


class Scheduler:
    """
    Adapter for integrating ScheduleFlow into RAPS.

    This scheduler implements the same interface as the default RAPS scheduler.
    It converts RAPS jobs into ScheduleFlow’s format, calls ScheduleFlow’s scheduling
    routines, then updates the RAPS job objects accordingly.
    """

    def __init__(self, config, policy, bfpolicy, resource_manager, jobs):
        self.sorted_priorities = sorted([x.priority for x in jobs])
        num_prios = len(self.sorted_priorities)
        # self.sf_queue = []
        self.queue = []  # track submitted jobs
        self.config = config
        self.policy = policy
        self.bfpolicy = bfpolicy
        self.resource_manager = resource_manager
        self.sf_scheduler = ScheduleFlow.Scheduler(
            ScheduleFlow.System(config['TOTAL_NODES']),
            priorityLevels=num_prios,
        )
        self._sf_runtime = _intScheduleFlow.Runtime([])
        self._sf_runtime.scheduler = self.sf_scheduler
        # self.sf_time = -1
        self.sf_submitted_list = []  # list of sf_apps
        # self.sf_start_list = []  # list as returned from sf_scheduler.submit_job
        # self.sf_end_list = []  # list as returned from sf_scheduler.start_job
        # self.sf_action_list = []  # list as returned from sf_scheduler.stop_job

    def gif(self):
        # logs = self._sf_runtime.get_stats()  # Unused
        # vis_hanlder = _intScheduleFlow.VizualizationEngine(self.sf_scheduler.
        self._sf_runtime._Runtime__generate_gif()

    def sort_jobs(self, queue, accounts=None):
        """
        Optionally, pre-sort jobs.

        For now, we can sort by submit_time (FCFS) as a default.
        """
        return sorted(queue, key=lambda job: job.submit_time)

    def start_job_event():
        pass

    def end_job_event():
        pass

    def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):

        # self._sf_runtim
        pass
        # SECOND TRY
        new_queue_items = list(filter(lambda x: x not in self.queue, queue))
        if new_queue_items:
            self.queue += new_queue_items
        #    # Convert RAPS jobs to ScheduleFlow format
            new_sf_jobs = [self._convert_to_sf(job) for job in new_queue_items]
            self.sf_submitted_list += new_sf_jobs  # This one only holds sf_jobs no timestamps
            # Submit each job to the ScheduleFlow scheduler # This trigger schedule!
            if new_sf_jobs:
                ret = self.sf_scheduler.submit_job(current_time, new_sf_jobs)
                self._sf_runtime._Runtime__handle_scheduler_actions(ret)
                self._sf_runtime._Runtime__trigger_schedule_event()

        if not self._sf_runtime._Runtime__events.empty():
            top = self._sf_runtime._Runtime__events.top()
            if top[0] == current_time:
                start_jobs = []
                end_jobs = []
                for event in self._sf_runtime._Runtime__events.pop_list():
                    if event[1] == EventType.Submit:
                        raise ValueError(f"Didnt we already Submit above? {event}")
                    if event[1] == EventType.JobStart:
                        start_jobs.append(event[2])
                    if event[1] == EventType.JobEnd:
                        end_jobs.append(event[2])
                if len(end_jobs) > 0:
                    self._sf_runtime._Runtime__job_end_event(end_jobs)
                    # End of jobs is handled by RAPS via prepare_timestep
                    pass
                if len(start_jobs) > 0:
                    self._sf_runtime._Runtime__job_start_event(start_jobs)
                    for sf_app in start_jobs:
                        job = _match_sf_app_and_job(sf_app, queue, start_jobs)
                        queue.remove(job)
                        self.resource_manager.assign_nodes_to_job(job, current_time, self.policy)
                        running.append(job)

            # Keep track of:  All jobs have been submitted empty the queue!

        #    remove_list = []
        #    job_list = []
        #    for x in self.sf_start_list:
        #        sf_job_start_time,sf_app = x
        #        if sf_job_start_time <= current_time:
        #            job_list.append(sf_app)
        #            remove_list.append(x)
        #            job = _match_sf_app_and_job(sf_app,queue,self.sf_submitted_list)
        #            if current_time != sf_job_start_time:
        #                print("current_time != sf_job_start_time")
        #                print(f"{current_time} != {sf_job_start_time}")
        #            queue.remove(job)
        #            self.sf_submitted_list.remove(sf_app)

        #            self.resource_manager.assign_nodes_to_job(job, current_time)
        #            running.append(job)
        #    if job_list:
        #        self.sf_end_list += self.sf_scheduler.start_job(current_time,job_list)
        #    for x in remove_list:
        #        self.sf_start_list.remove(x)

        # First TRY
        # if self.sf_end_list:
        #    remove_list = []
        #    job_list = []
        #    for x in self.sf_end_list:
        #        if x[0] <= current_time:
        #            job_list.append(x[1])
        #            remove_list.append(x)
        #    if job_list:
        #        self.sf_action_list += self.sf_scheduler.stop_job(current_time,job_list)
        #    for x in remove_list:
        #        self.sf_end_list.remove(x)

        # submit_jobs triggered the schedule calculation, sf_jobs returned the placed jobs.
        # We need to flect this on the raps side.

        # March the sf_scheduler forward based on the jobs
        # end_jobs = self.sf_scheduler.start_job(current_time,sf_schedule[1])
        # self.sf_scheduler.end_job(current_time,end_jobs)

        # Add to running

        # Process the actions (each action is assumed to be (start_time, job_info))
        # for act in actions:
        #    start_time, sf_job = act
        #    # Find the corresponding RAPS job using its ID
        #    job = self._find_job(queue, sf_job['job_id'])
        #    if job:
        #        job.scheduled_nodes = sf_job.get('assigned_nodes', [])
        #        job.start_time = start_time
        #        job.end_time = start_time + job.wall_time
        #        job.state = JobState.RUNNING
        #        running.append(job)
        #        queue.remove(job)
        #        if debug:
        #            print(f"t={current_time}: Scheduled job {job.id} on nodes {summarize_ranges(job.scheduled_nodes)}")

    def _find_sf_in_queue(self, queue, sf_app):
        # Remember we added four digits and an underscore in _convert_to_sf:
        match = [x for x in queue if x.id == sf_app.name]
        if len(match != 1):
            raise ValueError(sf_app)
        return match[0]

    def _convert_to_sf(self, job):
        # Create an ScheduleFlow.Application from the job information:
        sf_prio = self.sorted_priorities.index(job.priority)
        # Use job_dict to create a dictionary from the RAPS job.
        nodes = job.nodes_required
        submission_time = job.submit_time
        if submission_time < 0:
            submission_time = 0
        walltime = job.wall_time
        requested_walltimes = [job.wall_time]
        priority = sf_prio
        resubmit_factor = -1
        name = job.id  # We use the ID as name to be able to match when unpacking!
        return ScheduleFlow.Application(nodes,
                                        submission_time,
                                        walltime,
                                        requested_walltimes,
                                        priority,
                                        resubmit_factor,
                                        name)

    def _find_job(self, queue, job_id):
        """
        Find the RAPS job in the queue that matches the given job_id.
        """
        for job in queue:
            if job.job_id == job_id:
                return job
        return None

    def find_backfill_job(self, queue, num_free_nodes, current_time):
        """
        Optionally, implement backfill logic by delegating to ScheduleFlow's
        mechanisms or by applying custom logic.
        """
        # This is left as an exercise. You might use ScheduleFlow’s API to determine if a job can backfill.
        return None


def _match_sf_app_and_job(sf_app, queue, sf_queue):
    match = [x for x in sf_queue if x.name == sf_app.name]
    if len(match) != 1:
        print("Multiple Matches")
        raise ValueError(sf_app)
    else:
        match = match[0]
    job = [x for x in queue if x.id == match.name]
    if len(job) != 1:
        print("Multiple submitted Jobs ")
        raise ValueError(job)
    else:
        job = job[0]
    return job


if __name__ == '__main__':
    import unittest
    unittest.main()
