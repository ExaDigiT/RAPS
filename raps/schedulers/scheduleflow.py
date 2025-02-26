from raps.job import Job, JobState
from raps.utils import summarize_ranges
from third_party.ScheduleFlow import ScheduleFlow
from ..job import job_dict

class SFJob:
    def __init__(self, job_info):
        """Map RAPS job object to ScheduleFlow"""
        self.job_id = job_info['id']
        self.nodes = job_info['nodes_required']
        self.walltime = job_info['wall_time']
        self.requested_walltimes = None
        self.submission_time = job_info['submit_time']
        self.name = job_info['name']
        self.priority = job_info['priority']
        self.resubmit_factor = -1

    def __hash__(self):
        return hash(self.job_id)

    def __eq__(self, other):
        return isinstance(other, SFJob) and self.id == other.id

    def __repr__(self):
        return f"SFJob(id={self.job_id}, nodes={self.nodes}, wall_time={self.walltime})"

class Scheduler:
    """
    Adapter for integrating ScheduleFlow into RAPS.
    
    This scheduler implements the same interface as the default RAPS scheduler.
    It converts RAPS jobs into ScheduleFlow’s format, calls ScheduleFlow’s scheduling
    routines, then updates the RAPS job objects accordingly.
    """

    def __init__(self, config, policy, resource_manager):
        self.config = config
        self.policy = policy  
        self.resource_manager = resource_manager
        self.sf_scheduler = ScheduleFlow.Scheduler(
            ScheduleFlow.System(config['TOTAL_NODES']),
            priorityLevels=3,
        )

    def sort_jobs(self, queue, accounts=None):
        """
        Optionally, pre-sort jobs.
        
        For now, we can sort by submit_time (FCFS) as a default.
        """
        return sorted(queue, key=lambda job: job.submit_time)

    def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
        # Convert RAPS jobs to ScheduleFlow format
        sf_jobs = [self._convert_to_sf(job) for job in queue]
        
        # Submit each job to the ScheduleFlow scheduler
        for sf_job in sf_jobs:
            self.sf_scheduler.submit_job(current_time, [sf_job])
        
        # Trigger the schedule calculation
        actions = self.sf_scheduler.trigger_schedule(current_time)
        
        # Process the actions (each action is assumed to be (start_time, job_info))
        for act in actions:
            start_time, sf_job = act
            # Find the corresponding RAPS job using its ID
            job = self._find_job(queue, sf_job['id'])
            if job:
                job.scheduled_nodes = sf_job.get('assigned_nodes', [])
                job.start_time = start_time
                job.end_time = start_time + job.wall_time
                job.state = JobState.RUNNING
                running.append(job)
                queue.remove(job)
                if debug:
                    print(f"t={current_time}: Scheduled job {job.id} on nodes {summarize_ranges(job.scheduled_nodes)}")

    def _convert_to_sf(self, job):
        # Use job_dict to create a dictionary from the RAPS job.
        d = job_dict(
            job.nodes_required,
            job.name,
            job.account,
            job.cpu_trace,
            job.gpu_trace,
            job.ntx_trace,
            job.nrx_trace,
            job.wall_time,
            getattr(job, 'end_state', None),  # Provide a default if not set
            job.requested_nodes,
            job.submit_time,
            job.id,
            priority=job.priority,
            partition=getattr(job, 'partition', 0)
        )
        # Now create an SFJob from the dictionary.
        return SFJob(d)

    def _find_job(self, queue, job_id):
        """
        Find the RAPS job in the queue that matches the given job_id.
        """
        for job in queue:
            if job.id == job_id:
                return job
        return None

    def find_backfill_job(self, queue, num_free_nodes, current_time):
        """
        Optionally, implement backfill logic by delegating to ScheduleFlow's
        mechanisms or by applying custom logic.
        """
        # This is left as an exercise. You might use ScheduleFlow’s API to determine if a job can backfill.
        return None

if __name__ == '__main__':
    import unittest
    unittest.main()
