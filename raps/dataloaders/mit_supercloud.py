
import numpy as np
import os
from raps.job import job_dict

def load_data(file_path, **kwargs):
    """
    Loads MIT Supercloud data from a pickled file and transforms it into a list of job dictionaries.

    Args:
        file_path (str): The path to the pickled data file.

    Returns:
        list: A list of job dictionaries.
    """
    with np.load(file_path, allow_pickle=True) as data:
        jobs_data = data['jobs']

    jobs = []
    for job_dict_data in jobs_data:
        # Convert numpy.ndarray to list for cpu_trace and gpu_trace if they are arrays
        cpu_trace = job_dict_data.item().get('cpu_trace', [])
        if isinstance(cpu_trace, np.ndarray):
            cpu_trace = cpu_trace.tolist()

        gpu_trace = job_dict_data.item().get('gpu_trace', [])
        if isinstance(gpu_trace, np.ndarray):
            gpu_trace = gpu_trace.tolist()

        job = job_dict(
            id=job_dict_data.item().get('id'),
            name=job_dict_data.item().get('name'),
            account=job_dict_data.item().get('account'),
            nodes_required=job_dict_data.item().get('nodes_required'),
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            ntx_trace=job_dict_data.item().get('ntx_trace', []),
            nrx_trace=job_dict_data.item().get('nrx_trace', []),
            end_state=job_dict_data.item().get('end_state'),
            submit_time=job_dict_data.item().get('submit_time'),
            time_limit=job_dict_data.item().get('time_limit'),
            start_time=job_dict_data.item().get('start_time'),
            end_time=job_dict_data.item().get('end_time'),
            wall_time=job_dict_data.item().get('wall_time'),
            trace_time=job_dict_data.item().get('trace_time', 0),
            trace_start_time=job_dict_data.item().get('trace_start_time', 0),
            trace_end_time=job_dict_data.item().get('trace_end_time', 0),
            trace_missing_values=job_dict_data.item().get('trace_missing_values', False)
        )
        jobs.append(job)

    return jobs
