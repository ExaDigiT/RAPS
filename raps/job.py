from enum import Enum
import numpy as np
from types import NoneType

"""
Note: want to simplify this in the future to use a minimal required set of job attributes,
the standard workload format (swf) https://www.cs.huji.ac.il/labs/parallel/workload/swf.html

Implementing such using something like:

    from types import SimpleNamespace
    job = SimpleNamespace(**job_dict(...))
"""


class JobState(Enum):
    """Enumeration for job states."""
    RUNNING = 'R'
    PENDING = 'PD'
    COMPLETED = 'C'
    COMPLETING = 'Cing'
    CANCELLED = 'CA'
    FAILED = 'F'
    TIMEOUT = 'TO'


def job_dict(*,
             nodes_required,
             name,
             account,
             # Allocation
             current_state=JobState.PENDING,
             end_state: JobState | None = None,
             scheduled_nodes=None,
             id,
             priority: int | None = 0,
             partition: int | None = 0,
             # Resource Requests and allocations
             cpu_cores_required=0,
             gpu_units_required=0,
             allocated_cpu_cores=0,
             allocated_gpu_units=0,
             # Traces
             cpu_trace,
             gpu_trace,
             ntx_trace,
             nrx_trace,
             # Times
             submit_time=0,
             time_limit: int = 0,
             start_time: int | None = 0,
             end_time: int | None = 0,
             expected_run_time: int | None = 0,
             current_run_time: int = 0,
             trace_time: int | None = 0,
             trace_start_time: int | None = 0,
             trace_end_time: int | None = 0,
             trace_quanta: int | None = None,
             trace_missing_values: bool | None = False,
             downscale: int = 1
             ):
    """ Return job info dictionary """
    return {
        'nodes_required': nodes_required,
        'name': name,
        'account': account,
        # Allocation:
        'current_state': current_state,
        'end_state': end_state,
        'scheduled_nodes': scheduled_nodes,
        'id': id,
        'priority': priority,
        'partition': partition,
        # Resource Requests and allocations:
        'cpu_cores_required': cpu_cores_required,
        'gpu_units_required': gpu_units_required,
        'allocated_cpu_cores': allocated_cpu_cores,
        'allocated_gpu_units': allocated_gpu_units,
        # Traces:
        'cpu_trace': cpu_trace,
        'gpu_trace': gpu_trace,
        'ntx_trace': ntx_trace,
        'nrx_trace': nrx_trace,
        # Times:
        'submit_time': submit_time,
        'time_limit': time_limit,
        'start_time': start_time,
        'end_time': end_time,
        'expected_run_time': expected_run_time,
        'current_run_time': current_run_time,
        'trace_time': trace_time,
        'trace_start_time': trace_start_time,
        'trace_end_time': trace_end_time,
        'trace_quanta': trace_quanta,
        'trace_missing_values': trace_missing_values,
        'dilated': False,
        'downscale': downscale
    }


def dilate_trace(trace, factor):
    """
    Scale a trace in the time dimension by the given factor.

    Parameters:
    - trace: list/tuple/np.ndarray of floats OR a single numeric scalar.
    - factor (float): >1 to slow down (stretch in time), <1 to speed up.

    Returns:
    - list of float for sequence inputs, or numeric for scalar inputs.
    """
    if trace is None:
        return trace

    if factor is None:
        raise ValueError("factor must be provided")
    if factor == 0:
        raise ValueError("factor must be non-zero")

    # Treat any numeric scalar (int/float/np.number) as a scalar trace
    if isinstance(trace, (int, float, np.integer, np.floating, np.number)):
        # Keep total "area" the same when stretching/compressing in time:
        return trace / factor

    # Handle common sequence types directly
    if isinstance(trace, (list, tuple, np.ndarray)):
        arr = np.asarray(trace, dtype=float)
    else:
        # Last-resort: try coercion (e.g., pandas Series)
        arr = np.asarray(trace, dtype=float)

    if arr.size == 0:
        # empty sequence: nothing to do
        return [] if not isinstance(trace, np.ndarray) else arr

    original_length = arr.size
    # at least 1 sample after dilation
    new_length = max(1, int(np.round(original_length * float(factor))))

    # If original_length == 1, interpolation just repeats the value
    old_indices = np.linspace(0, original_length - 1, num=original_length)
    new_indices = np.linspace(0, original_length - 1, num=new_length)

    new_trace = np.interp(new_indices, old_indices, arr).tolist()
    return new_trace


class Job:
    """Represents a job to be scheduled and executed in the distributed computing system.

    Each job consists of various attributes such as the number of nodes required for execution,
    CPU and GPU utilization, trace time, and other relevant parameters (see utils.job_dict).
    The job can transition through different states during its lifecycle, including PENDING,
    RUNNING, COMPLETED, CANCELLED, FAILED, or TIMEOUT.
    """
    _id_counter = 0

    def __init__(self, job_dict, current_state=JobState.PENDING, end_state=None, account=None):
        # # current_time unused!
        # Initializations:
        self.power = 0
        self.scheduled_nodes = []  # Explicit list of requested nodes
        self.nodes_required = 0  # If scheduled_nodes is set this can be derived.
        self.cpu_cores_required = 0
        self.gpu_units_required = 0
        self.allocated_cpu_cores = 0
        self.allocated_gpu_units = 0
        self.power_history = []
        self._current_state = current_state
        self.end_state = end_state  # default None!
        self.account = account
        # Times:
        self.submit_time = None   # Actual submit time
        self.time_limit = None    # Time limit set at submission
        self.start_time = None    # Actual start time when executing or from telemetry
        self.end_time = None      # Actual end time, either None if or from telemetry
        self.expected_run_time = None
        self.current_run_time = 0
        self.trace_time = None    # Time period for which traces are available
        self.trace_start_time = None  # Relative start time of the trace (to running time)
        self.trace_end_time = None    # Relative end time of the trace
        self.trace_quanta = None  # Trace quanta associated with the job # None means single value!
        self.running_time = 0     # Current running time updated when simulating

        # If a job dict was given, override the values from the job_dict:
        for key, value in job_dict.items():
            setattr(self, key, value)
        # In any case: provide a job_id!
        if self.id is None:  # This is wrong
            self.id = Job._get_next_id()

        if self.nodes_required == 0 and self.scheduled_nodes != []:
            self.nodes_required = len(self.scheduled_nodes)
        elif self.nodes_required != 0:
            pass
        else:
            raise ValueError(f"{self.nodes_required} {self.scheduled_nodes}")
        if self.scheduled_nodes == [] or self.scheduled_nodes is None or \
           (isinstance(self.scheduled_nodes, list) and isinstance(self.scheduled_nodes[0], int)) or \
           (isinstance(self.scheduled_nodes, np.ndarray) and isinstance(self.scheduled_nodes[0], int)):
            pass  # Type is ok
        else:
            raise ValueError(
                    f"type: self.scheduled_nodes:{type(self.scheduled_nodes)}, "
                    f"with {type(self.scheduled_nodes[0])}")
        assert isinstance(self.submit_time, (int, float))
        assert isinstance(self.expected_run_time, (int, float, np.int64, np.double, NoneType))
        assert isinstance(self.current_run_time, (int, float, np.int64, np.double))
        assert isinstance(self.start_time, (int, float, np.int64, np.double, NoneType))
        assert isinstance(self.end_time, (int, float, np.int64, np.double, NoneType))
        if self.start_time is not None and self.end_time is not None:
            assert self.start_time <= self.end_time, f"{self.start_time} <= {self.end_time}"

    def __repr__(self):
        """Return a string representation of the job."""
        return (f"Job(id={self.id}, name={self.name}, account={self.account}, "
                f"nodes_required={self.nodes_required}, "
                f"scheduled_nodes={self.scheduled_nodes},  "
                f"cpu_cores_required={self.cpu_cores_required}, "
                f"gpu_units_required={self.gpu_units_required}, "
                f"allocated_cpu_cores={self.allocated_cpu_cores}, "
                f"allocated_gpu_units={self.allocated_gpu_units}, "
                f"cpu_trace={self.cpu_trace}, gpu_trace={self.gpu_trace}, "
                f"ntx_trace={self.ntx_trace}, nrx_trace={self.nrx_trace}, "
                f"end_state={self.end_state}, "
                f"current_state={self.current_state}, "
                f"submit_time={self.submit_time}, time_limit={self.time_limit}, "
                f"start_time={self.start_time}, end_time={self.end_time}, "
                f"expected_run_time={self.expected_run_time}, "
                f"current_run_time={self.current_run_time}, "
                f"trace_time={self.trace_time}, "
                f"trace_start_time={self.trace_start_time}, "
                f"trace_end_time={self.trace_end_time}, "
                f"trace_quanta={self.trace_quanta}, "
                f"running_time={self.running_time}, "
                f"power={self.power}, "
                f"power_history={self.power_history})")

    @property
    def current_state(self):
        """Get the current state of the job."""
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        """Set the current_state of the job."""
        if isinstance(value, JobState):
            self._current_state = value
        elif isinstance(value, str) and value in JobState.__members__:
            self._current_state = JobState[value]
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
        Apply a dilation factor to the job’s execution traces and run time.

        Parameters:
        - factor (float): the dilation factor; >1 to slow down (lengthen the traces) and <1 to speed up.
        """
        self.cpu_trace = dilate_trace(self.cpu_trace, factor)
        self.gpu_trace = dilate_trace(self.gpu_trace, factor)
        self.ntx_trace = dilate_trace(self.ntx_trace, factor)
        self.nrx_trace = dilate_trace(self.nrx_trace, factor)
        if self.end_time is not None:
            expected_run_time = self.end_time - self.start_time
            expected_run_time = int(np.round(expected_run_time * factor))
            assert self.start_time is not None
            self.end_time = self.start_time + expected_run_time


class JobStatistics:
    """ Reduced class for handling statistics after the job has finished.  """

    def __init__(self, job):
        self.id = job.id
        self.name = job.name
        self.account = job.account
        self.num_nodes = len(job.scheduled_nodes)
        self.scheduled_nodes = job.scheduled_nodes
        self.run_time = job.running_time
        self.submit_time = job.submit_time
        self.start_time = job.start_time
        self.end_time = job.end_time
        self.current_state = job.current_state
        if isinstance(job.cpu_trace, list) or isinstance(job.cpu_trace, np.ndarray):
            if len(job.cpu_trace) == 0:
                self.avg_cpu_usage = 0
            else:
                self.avg_cpu_usage = sum(job.cpu_trace) / len(job.cpu_trace)
        elif isinstance(job.cpu_trace, int) or isinstance(job.cpu_trace, float):
            self.avg_cpu_usage = job.cpu_trace
        elif job.cpu_trace is None:
            self.avg_cpu_usage = None
        else:
            raise NotImplementedError()

        if isinstance(job.gpu_trace, list) or isinstance(job.gpu_trace, np.ndarray):
            if len(job.gpu_trace) == 0:
                self.avg_gpu_usage = 0
            else:
                self.avg_gpu_usage = sum(job.gpu_trace) / len(job.gpu_trace)
        elif isinstance(job.gpu_trace, int) or isinstance(job.gpu_trace, float):
            self.avg_gpu_usage = job.gpu_trace
        elif job.gpu_trace is None:
            self.avg_gpu_usage = None
        else:
            raise NotImplementedError()

        if isinstance(job.ntx_trace, list) or isinstance(job.ntx_trace, np.ndarray):
            if len(job.ntx_trace) == 0:
                self.avg_ntx_usage = 0
            else:
                self.avg_ntx_usage = sum(job.ntx_trace) / len(job.ntx_trace)
        elif isinstance(job.ntx_trace, int) or isinstance(job.ntx_trace, float):
            self.avg_ntx_usage = job.ntx_trace
        elif job.ntx_trace is None:
            self.avg_ntx_usage = None
        else:
            raise NotImplementedError()

        if isinstance(job.nrx_trace, list) or isinstance(job.nrx_trace, np.ndarray):
            if len(job.nrx_trace) == 0:
                self.avg_nrx_usage = 0
            else:
                self.avg_nrx_usage = sum(job.nrx_trace) / len(job.nrx_trace)
        elif isinstance(job.nrx_trace, int) or isinstance(job.nrx_trace, float):
            self.avg_nrx_usage = job.nrx_trace
        elif job.nrx_trace is None:
            self.avg_nrx_usage = None
        else:
            raise NotImplementedError()

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
    expected_run_time = 600    # total job run time in seconds (600s = 10 minutes)
    num_samples = expected_run_time // trace_quanta  # should be 40 samples

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
        expected_run_time=expected_run_time,
        end_state="",
        scheduled_nodes=[],
        time_offset=0,
        job_id=0
    )

    # Instantiate the Job.
    job_instance = Job(jdict, current_time=0)

    # Print original job properties.
    print("Original expected_run_time:", job_instance.expected_run_time)
    print("Original cpu_trace length:", len(job_instance.cpu_trace))
    print("Original gpu_trace length:", len(job_instance.gpu_trace))

    # Apply a dilation factor, e.g., 1.5 for a 50% slowdown (traces become 50% longer)
    dilation_factor = 1.5
    job_instance.apply_dilation(dilation_factor)

    # Calculate the expected new lengths.
    expected_samples = int(np.round(num_samples * dilation_factor))
    expected_run_time = int(np.round(expected_run_time * dilation_factor))

    # Print the dilated job properties.
    print("\nAfter applying a dilation factor of", dilation_factor)
    print("New expected_run_time:", job_instance.expected_run_time, "(expected:", expected_run_time, ")")
    print("New cpu_trace length:", len(job_instance.cpu_trace), "(expected:", expected_samples, ")")
    print("New gpu_trace length:", len(job_instance.gpu_trace), "(expected:", expected_samples, ")")

    # Optionally, print a few sample values from the new traces.
    print("\nSample cpu_trace values:", job_instance.cpu_trace[:5])
    print("Sample gpu_trace values:", job_instance.gpu_trace[:5])
