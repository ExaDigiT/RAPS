from raps.job import Job, JobState
from raps.utils import summarize_ranges
# Import ScheduleFlow’s modules – since ScheduleFlow isn’t pip installable, you
# may have vendored it or added it as a submodule (e.g. under third_party/scheduleflow)
from third_party.ScheduleFlow import ScheduleFlow  # adjust this import if needed

class Scheduler:
    """
    Adapter for integrating ScheduleFlow into RAPS.
    
    This scheduler implements the same interface as the default RAPS scheduler.
    It converts RAPS jobs into ScheduleFlow’s format, calls ScheduleFlow’s scheduling
    routines, then updates the RAPS job objects accordingly.
    """

    def __init__(self, config, policy, resource_manager):
        self.config = config
        # You might or might not use the policy parameter; for now we store it.
        self.policy = policy  
        self.resource_manager = resource_manager
        # Here we instantiate a ScheduleFlow scheduler.
        # For example, if ScheduleFlow provides a Scheduler or OnlineScheduler,
        # choose one based on your needs. (See ScheduleFlow documentation for details.)
        self.sf_scheduler = ScheduleFlow.Scheduler(
            ScheduleFlow.System(config['TOTAL_NODES']),
            # You might pass additional parameters here if needed.
        )

    def sort_jobs(self, queue, accounts=None):
        """
        Optionally, pre-sort jobs.
        
        For now, we can sort by submit_time (FCFS) as a default.
        """
        return sorted(queue, key=lambda job: job.submit_time)

    def schedule(self, queue, running, current_time, accounts=None, sorted=False, debug=False):
        """
        Convert the list of RAPS jobs into the format ScheduleFlow expects,
        call ScheduleFlow’s scheduling function, and then update each job.

        This method is expected to remove the scheduled jobs from `queue` and append them to `running`.
        """
        # Convert RAPS jobs into ScheduleFlow job representations.
        sf_jobs = [self._convert_job(job) for job in queue]

        # Call ScheduleFlow’s scheduling algorithm.
        # This is a placeholder – you must adapt it to ScheduleFlow’s actual API.
        scheduled_sf_jobs = self.sf_scheduler.compute_schedule(sf_jobs)

        # Map ScheduleFlow’s output back to the corresponding RAPS jobs.
        # Here we assume each ScheduleFlow job has an 'id' and a field 'assigned_nodes'.
        for sf_job in scheduled_sf_jobs:
            job = self._find_job_by_id(queue, sf_job['id'])
            if job is not None:
                job.scheduled_nodes = sf_job.get('assigned_nodes', [])
                # You could also update start_time, end_time, etc., if ScheduleFlow provides these.
                job.start_time = current_time  # Or use sf_job['start_time'] if available
                job.end_time = current_time + job.wall_time
                job.state = JobState.RUNNING
                running.append(job)
                queue.remove(job)
                if debug:
                    print(f"t={current_time}: Scheduled job {job.id} on nodes {summarize_ranges(job.scheduled_nodes)}")
        # Optionally, if ScheduleFlow supports backfill, you can implement find_backfill_job() similarly.

    def _convert_job(self, job):
        """
        Convert a RAPS Job object into a dictionary (or other format) that ScheduleFlow expects.
        
        Adjust the fields as necessary – here’s an example conversion.
        """
        return {
            'id': job.id,
            'nodes_required': job.nodes_required,
            'wall_time': job.wall_time,
            'submit_time': job.submit_time,
            # Add any additional fields required by ScheduleFlow here.
        }

    def _find_job_by_id(self, queue, job_id):
        """
        Given a list of RAPS jobs, return the one with the matching id.
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
