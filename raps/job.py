from enum import Enum

def job_dict(nodes_required, name, cpu_trace, gpu_trace, ntx_trace, nrx_trace, \
             wall_time, end_state, scheduled_nodes, time_offset, job_id, priority=0):
    """ Return job info dictionary """
    return {
        'nodes_required': nodes_required,
        'name': name,
        'cpu_trace': cpu_trace,
        'gpu_trace': gpu_trace,
        'ntx_trace': ntx_trace,
        'nrx_trace': nrx_trace,
        'wall_time': wall_time,
        'end_state': end_state,
        'requested_nodes': scheduled_nodes,
        'submit_time': time_offset,
        'id': job_id,
        'priority': priority
    }


class JobState(Enum):
    """Enumeration for job states."""
    RUNNING = 'R'
    PENDING = 'PD'
    COMPLETED = 'C'
    CANCELLED = 'CA'
    FAILED = 'F'
    TIMEOUT = 'TO'


class Job:
    """Represents a job to be scheduled and executed in the distributed computing system.

    Each job consists of various attributes such as the number of nodes required for execution,
    CPU and GPU utilization, wall time, and other relevant parameters (see utils.job_dict). 
    The job can transition through different states during its lifecycle, including PENDING, 
    RUNNING, COMPLETED, CANCELLED, FAILED, or TIMEOUT.
    """
    _id_counter = 0

    def __init__(self, job_dict, current_time, state=JobState.PENDING):
        for key, value in job_dict.items(): setattr(self, key, value)
        if not self.id: self.id = Job._get_next_id() 
        # initializations
        self.start_time = None
        self.end_time = None
        self.running_time = 0
        self.power = 0
        self.scheduled_nodes = []
        self.power_history = [] 
        self._state = state

    def __repr__(self):
        """Return a string representation of the job."""
        return (f"Job(id={self.id}, name={self.name}, nodes_required={self.nodes_required}, "
                f"cpu_trace={self.cpu_trace}, gpu_trace={self.gpu_trace}, wall_time={self.wall_time}, "
                f"end_state={self.end_state}, requested_nodes={self.requested_nodes}, "
                f"submit_time={self.submit_time}, start_time={self.start_time}, "
                f"end_time={self.end_time}, running_time={self.running_time}, state={self._state}, "
                f"scheduled_nodes={self.scheduled_nodes}, power={self.power}, "
                f"power_history={self.power_history})")

    @property
    def state(self):
        """Get the current state of the job."""
        return self._state

    @state.setter
    def state(self, value):
        """Set the state of the job."""
        if isinstance(value, JobState):
            self._state = value
        elif isinstance(value, str) and value in JobState.__members__:
            self._state = JobState[value]
        else:
            raise ValueError(f"Invalid state: {value}")

    @classmethod
    def _get_next_id(cls):
        """Generate the next unique identifier for a job.

        This method is used internally to generate a unique identifier for each job
        based on the current value of the class's _id_counter attribute. Each time
        this method is called, it increments the counter by 1 and returns the new value.

        Returns:
        - int: The next unique identifier for a job.
        """
        cls._id_counter += 1
        return cls._id_counter
