import numpy as np
from enum import Enum
import numpy as np

"""
Note: want to simplify this in the future to use a minimal required set of job attributes,
the standard workload format (swf) https://www.cs.huji.ac.il/labs/parallel/workload/swf.html

Implementing such using something like:

    from types import SimpleNamespace
    job = SimpleNamespace(**job_dict(...))
"""


def job_dict(*, nodes_required, name, account,
             cpu_trace, gpu_trace, ntx_trace, nrx_trace,
             end_state, scheduled_nodes=None, id, priority=0, partition=0,
             submit_time=0, time_limit=0, start_time=0, end_time=0,
             wall_time=0, trace_time=0, trace_start_time=0, trace_end_time=0,
             trace_missing_values=False):
    """ Return job info dictionary """
    return {
        'nodes_required': nodes_required,
        'name': name,
        'account': account,
        'cpu_trace': cpu_trace,
        'gpu_trace': gpu_trace,
        'ntx_trace': ntx_trace,
        'nrx_trace': nrx_trace,
        'end_state': end_state,
        'requested_nodes': scheduled_nodes,
        'id': id,
        'priority': priority,
        'partition': partition,
        # Times:
        'submit_time': submit_time,
        'time_limit': time_limit,
        'start_time': start_time,
        'end_time': end_time,
        'wall_time': wall_time,
        'trace_time': trace_time,
        'trace_start_time': trace_start_time,
        'trace_end_time': trace_end_time,
        'trace_missing_values': trace_missing_values,
        'dilated': False
    }


def dilate_trace(trace, factor):
    """
    Scale a trace in the time dimension by the given factor.

    Parameters:
    - trace (list of float): the original trace values.
    - factor (float): the dilation factor; >1 to slow down (stretch) and <1 to speed up (compress).

    Returns:
    - list of float: the dilated trace.
    """
    if trace is None or len(trace) == 0:
        return trace
    original_length = len(trace)
    # Compute the new length (rounding to the nearest integer)
    new_length = int(np.round(original_length * factor))
    # Create arrays for the old and new indices
    old_indices = np.linspace(0, original_length - 1, num=original_length)
    new_indices = np.linspace(0, original_length - 1, num=new_length)
    # Use linear interpolation to compute the new trace values
    new_trace = np.interp(new_indices, old_indices, trace).tolist()
    return new_trace


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
    CPU and GPU utilization, trace time, and other relevant parameters (see utils.job_dict).
    The job can transition through different states during its lifecycle, including PENDING,
    RUNNING, COMPLETED, CANCELLED, FAILED, or TIMEOUT.
    """
    _id_counter = 0

    def __init__(self, job_dict, state=JobState.PENDING, account=None):
        # # current_time unused!
        # Initializations:
        self.power = 0
        self.scheduled_nodes = []  # Explicit list of requested nodes
        self.nodes_required = 0  # If scheduled_nodes is set this can be derived.
        self.power_history = []
        self._state = state
        self.account = account
        # Times:
        self.submit_time = None   # Actual submit time
        self.time_limit = None    # Time limit set at submission
        self.start_time = None    # Actual start time when executing or from telemetry
        self.end_time = None      # Actual end time when executing or from telemetry
        self.wall_time = None     # end_time - start_time
        self.trace_time = None    # Time period for which traces are available
        self.trace_start_time = None  # Relative start time of the trace (to running time)
        self.trace_end_time = None    # Relative end time of the trace
        self.running_time = 0     # Current running time updated when simulating

        # If a job dict was given, override the values from the job_dict:
        for key, value in job_dict.items():
            setattr(self, key, value)
        # In any case: provide a job_id!
        if self.id is None:  # This is wrong
            self.id = Job._get_next_id()

        if self.scheduled_nodes and self.nodes_required == 0:
            self.nodes_required = len(self.scheduled_nodes)

    def __repr__(self):
        """Return a string representation of the job."""
        return (f"Job(id={self.id}, name={self.name}, account={self.account}, "
                f"nodes_required={self.nodes_required}, "
                f"cpu_trace={self.cpu_trace}, gpu_trace={self.gpu_trace}, "
                f"end_state={self.end_state}, requested_nodes={self.requested_nodes}, "
                f"submit_time={self.submit_time}, time_limit={self.time_limit}, "
                f"start_time={self.start_time}, end_time={self.end_time}, "
                f"wall_time={self.wall_time}, "
                f"trace_time={self.trace_time}, "
                f"trace_start_time={self.trace_start_time}, "
                f"trace_end_time={self.trace_end_time}, "
                f"running_time={self.running_time}, state={self._state}, "
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

    def statistics(self):
        """ Derive job statistics from the Job Class and return """
        return JobStatistics(self)

    def apply_dilation(self, factor):
        """
        Apply a dilation factor to the job’s execution traces and wall time.

        Parameters:
        - factor (float): the dilation factor; >1 to slow down (lengthen the traces) and <1 to speed up.
        """
        self.cpu_trace = dilate_trace(self.cpu_trace, factor)
        self.gpu_trace = dilate_trace(self.gpu_trace, factor)
        self.ntx_trace = dilate_trace(self.ntx_trace, factor)
        self.nrx_trace = dilate_trace(self.nrx_trace, factor)
        self.wall_time = int(np.round(self.wall_time * factor))
        self.end_time = self.start_time + self.wall_time



class JobStatistics:
    """ Reduced class for handling statistics after the job has finished.  """

    def __init__(self,job):
        self.id = job.id
        self.name = job.name
        self.account = job.account
        self.num_nodes = len(job.scheduled_nodes)
        self.run_time = job.running_time
        self.submit_time = job.submit_time
        self.start_time = job.start_time
        self.end_time = job.end_time
        self.state = job._state
        if isinstance(job.cpu_trace,list) or isinstance(job.cpu_trace,np.ndarray):
            if len(job.cpu_trace) == 0:
                self.avg_cpu_usage = 0
            else:
                self.avg_cpu_usage = sum(job.cpu_trace) / len(job.cpu_trace)
        elif isinstance(job.cpu_trace,int) or isinstance(job.cpu_trace,float):
            self.avg_cpu_usage = job.cpu_trace
        else:
            raise NotImplementedError()

        if isinstance(job.gpu_trace,list) or isinstance(job.gpu_trace,np.ndarray):
            if len(job.gpu_trace) == 0:
                self.avg_gpu_usage = 0
            else:
                self.avg_gpu_usage = sum(job.gpu_trace) / len(job.gpu_trace)
        elif isinstance(job.gpu_trace,int) or isinstance(job.gpu_trace,float):
            self.avg_gpu_usage = job.gpu_trace
        else:
            raise NotImplementedError()

        if isinstance(job.ntx_trace,list) or isinstance(job.ntx_trace,np.ndarray):
            if len(job.ntx_trace) == 0:
                self.avg_ntx_usage = 0
            else:
                self.avg_ntx_usage = sum(job.ntx_trace) / len(job.ntx_trace)
        elif isinstance(job.ntx_trace,int) or isinstance(job.ntx_trace,float):
            self.avg_ntx_usage = job.ntx_trace

        if isinstance(job.nrx_trace,list) or isinstance(job.nrx_trace,np.ndarray):
            if len(job.nrx_trace) == 0:
                self.avg_nrx_usage = 0
            else:
                self.avg_nrx_usage = sum(job.nrx_trace) / len(job.nrx_trace)
        elif isinstance(job.nrx_trace,int) or isinstance(job.nrx_trace,float):
            self.avg_nrx_usage = job.nrx_trace

        if len(job.power_history) == 0:
            self.avg_node_power = 0
            self.max_node_power = 0
        else:
            self.avg_node_power = sum(job.power_history) / len(job.power_history) / self.num_nodes
            self.max_node_power = max(job.power_history) / self.num_nodes
        self.energy = self.run_time * self.avg_node_power * self.num_nodes


if __name__ == "__main__":
    import random

    # Each sample in the trace represents 15 seconds.
    trace_quanta = 15  # seconds per sample
    wall_time = 600    # total job wall time in seconds (600s = 10 minutes)
    num_samples = wall_time // trace_quanta  # should be 40 samples

    # Generate a random GPU trace (values between 0 and 4 for 4 GPUs total)
    gpu_trace = [random.uniform(0, 4) for _ in range(num_samples)]
    # Generate a random CPU trace (values between 0 and 1)
    cpu_trace = [random.uniform(0, 1) for _ in range(num_samples)]
    # Dummy network traces
    ntx_trace = [random.uniform(0, 10) for _ in range(num_samples)]
    nrx_trace = [random.uniform(0, 10) for _ in range(num_samples)]

    # Create a job dictionary using the existing job_dict helper.
    jdict = job_dict(
        nodes_required=1,
        name="test_job",
        account="test_account",
        cpu_trace=cpu_trace,
        gpu_trace=gpu_trace,
        ntx_trace=ntx_trace,
        nrx_trace=nrx_trace,
        wall_time=wall_time,
        end_state="",
        scheduled_nodes=[],
        time_offset=0,
        job_id=0
    )

    # Instantiate the Job.
    job_instance = Job(jdict, current_time=0)

    # Print original job properties.
    print("Original wall_time:", job_instance.wall_time)
    print("Original cpu_trace length:", len(job_instance.cpu_trace))
    print("Original gpu_trace length:", len(job_instance.gpu_trace))

    # Apply a dilation factor, e.g., 1.5 for a 50% slowdown (traces become 50% longer)
    dilation_factor = 1.5
    job_instance.apply_dilation(dilation_factor)

    # Calculate the expected new lengths.
    expected_samples = int(np.round(num_samples * dilation_factor))
    expected_wall_time = int(np.round(wall_time * dilation_factor))

    # Print the dilated job properties.
    print("\nAfter applying a dilation factor of", dilation_factor)
    print("New wall_time:", job_instance.wall_time, "(expected:", expected_wall_time, ")")
    print("New cpu_trace length:", len(job_instance.cpu_trace), "(expected:", expected_samples, ")")
    print("New gpu_trace length:", len(job_instance.gpu_trace), "(expected:", expected_samples, ")")

    # Optionally, print a few sample values from the new traces.
    print("\nSample cpu_trace values:", job_instance.cpu_trace[:5])
    print("Sample gpu_trace values:", job_instance.gpu_trace[:5])
