import os
import re
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional, Generator, Any, Union

import numpy as np
import pandas as pd

from raps.job import job_dict, Job
from raps.utils import WorkloadData

"""
Official instructions are here:

https://drive.google.com/file/d/0B5g07T_gRDg9Z0lsSTEtTWtpOW8/view?resourcekey=0-cozD56gA4fUDdrkHnLJSrQ


---
Downloading Google Cluster Traces v2:

    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-471.0.0-darwin-arm.tar.gz
    tar -xf google-cloud-cli-471.0.0-darwin-arm.tar.gz
    gcloud components update

    gcloud auth login

    gsutil ls gs://clusterdata_2019_a/

    * collection_events
    * instance_events
    * instance_usage
    * machine_attributes
    * machine_events

    gsutil -m cp -r gs://clusterdata_2019_a/instance_usage-*.parquet.gz ./google_cluster_data/cell_a/instance_usage

    # Create a directory to store your sample data
    mkdir -p ./google_cluster_data_sample

    # Download the first JSON and Parquet file for collection_events
    gsutil cp gs://clusterdata_2019_a/collection_events-000000000000.json.gz ./google_cluster_data_sample/
    gsutil cp gs://clusterdata_2019_a/collection_events-000000000000.parquet.gz ./google_cluster_data_sample/

    # Download the first JSON and Parquet file for instance_events
    gsutil cp gs://clusterdata_2019_a/instance_events-000000000000.json.gz ./google_cluster_data_sample/
    gsutil cp gs://clusterdata_2019_a/instance_events-000000000000.parquet.gz ./google_cluster_data_sample/

    # Download the first JSON and Parquet file for instance_usage
    gsutil cp gs://clusterdata_2019_a/instance_usage-000000000000.json.gz ./google_cluster_data_sample/
    gsutil cp gs://clusterdata_2019_a/instance_usage-000000000000.parquet.gz ./google_cluster_data_sample/

    # ... and so on for other event types (machine_attributes, machine_events)
    gsutil cp gs://clusterdata_2019_a/machine_attributes-000000000000.json.gz ./google_cluster_data_sample/
    gsutil cp gs://clusterdata_2019_a/machine_attributes-000000000000.parquet.gz ./google_cluster_data_sample/

    gsutil cp gs://clusterdata_2019_a/machine_events-000000000000.json.gz ./google_cluster_data_sample/
    gsutil cp gs://clusterdata_2019_a/machine_events-000000000000.parquet.gz ./google_cluster_data_sample/

---
Following explanation from Gemini-CLI on how the job nodes required is being computed. Method must be verified

   1. Machine Capacity Determination:
       * The machine_events data is loaded to get information about the cluster's machines.
       * The CPU_capacity and memory_capacity of a typical machine are determined by taking
         the mode() (most frequent value) of these columns from the machine_df. This gives
         us the standard CPU and memory capacity of a single node in the cluster.

   2. Task Resource Request Aggregation:
       * The task_events data is loaded, which contains CPU_request and memory_request for
         individual tasks.
       * These task requests are then grouped by job_ID, and the CPU_request and memory_request
         are summed up for all tasks belonging to the same job. This gives us the total CPU and
         memory requested by each job.

   3. Nodes Required Calculation (CPU and Memory):
       * For each job, the total CPU_request is divided by the cpu_capacity of a single machine.
         The np.ceil() function is used to round up to the nearest whole number, ensuring that
         enough nodes are allocated to satisfy the CPU demand. This result is stored as
         nodes_required_cpu.
       * Similarly, the total memory_request is divided by the mem_capacity of a single machine,
         and np.ceil() is applied. This result is stored as nodes_required_mem.

   4. Final `nodes_required`:
       * The final nodes_required for a job is determined by taking the np.maximum() of nodes_required_cpu
         and nodes_required_mem. This ensures that the job is allocated enough nodes to satisfy both its CPU
         and memory requirements. The result is then cast to an integer (.astype(int)).

   5. Filtering:
       * Finally, any jobs for which the calculated nodes_required is 0 (meaning they requested no CPU or memory)
         are filtered out, as these jobs would not require any nodes in the simulation.
"""

# Define expected column names for each supported event type
V2_COLUMN_NAMES = {
    "job_events": [
        "timestamp",          # ↔ time
        "missing_info",       # ↔ missing_col_1
        "job_ID",
        "event_type",
        "user_name",
        "scheduling_class",
        "job_name",
        "logical_job_name"
    ],
    "machine_events": [
        "timestamp",
        "machine_ID",
        "event_type",
        "platform_ID",
        "CPU_capacity",
        "memory_capacity"
    ],
    "task_events": [
        "timestamp",
        "missing_info",
        "job_ID",
        "task_index",
        "machine_ID",
        "event_type",
        "user_name",
        "scheduling_class",
        "priority",
        "CPU_request",
        "memory_request",
        "disk_space_request",
        "different_machine_constraint"
    ],
    "task_usage": [
        "start_time",                        # file-col 0
        "end_time",                          # file-col 1
        "job_ID",                            # file-col 2
        "task_index",                        # file-col 3
        "machine_ID",                        # file-col 4
        "CPU_usage_rate",                    # file-col 5
        "memory_usage_avg",                  # file-col 6
        "memory_usage_max",                  # file-col 7
        "assigned_memory",                   # file-col 8
        "unmapped_page_cache_memory",        # file-col 9
        "page_cache_memory",                 # file-col 10
        "maximum_memory_usage",              # file-col 11
        "disk_IO_time_avg",                  # file-col 12
        "disk_IO_time_max",                  # file-col 13
        "local_disk_space_used",             # file-col 14
        "cycles_per_instruction",            # file-col 15
        "memory_accesses_per_instruction",   # file-col 16
        "sampling_rate",                     # file-col 17
        "aggregation_type",                  # file-col 18
        "missing_col_19"                     # file-col 19
    ]
}
SUPPORTED_EVENT_TYPES = list(V2_COLUMN_NAMES.keys())


class GoogleClusterV2DataLoader:
    """
    Loader for Google Cluster V2 CSV.GZ files.
    """

    def __init__(self, base_path: str, event_type: str = "job_events",
                 file_indices: Optional[List[int]] = None, concatenate: bool = True):
        self.base_path = os.path.expanduser(base_path)
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(f"Unsupported event type: '{event_type}'")
        self.event_type = event_type
        self.file_indices = file_indices
        self.concatenate = concatenate
        self.file_paths = self._find_files()

    def _find_files(self) -> List[str]:
        dir_path = os.path.join(self.base_path, self.event_type)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        files = os.listdir(dir_path)
        matches = []
        if self.file_indices:
            for idx in self.file_indices:
                pattern = re.compile(rf"part-{idx:05d}-of-\d{{5}}\.csv\.gz$")
                found = [f for f in files if pattern.match(f)]
                if not found:
                    raise FileNotFoundError(f"File index {idx} missing in {dir_path}")
                matches.extend(found)
        else:
            matches = [f for f in files if f.startswith("part-") and f.endswith(".csv.gz")]
        if not matches:
            raise FileNotFoundError(f"No files in {dir_path}")
        return [os.path.join(dir_path, f) for f in sorted(matches)]

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        dfs = []
        names = V2_COLUMN_NAMES[self.event_type]
        ts_col = names[0]
        for path in self.file_paths:
            df = pd.read_csv(path, compression='gzip', header=None,
                             names=names, dtype={ts_col: int})
            if not self.concatenate:
                yield df
            else:
                dfs.append(df)
        if self.concatenate and dfs:
            yield pd.concat(dfs, ignore_index=True)


def load_data(data_path: Union[str, List[str]], **kwargs: Any):
    config = kwargs.get('config')
    # Unpack list
    if isinstance(data_path, list):
        if len(data_path) == 1:
            data_path = data_path[0]
        else:
            raise ValueError(f"Expected single path, got {data_path}")
    base_path = os.path.expanduser(data_path)

    # Load machine events to determine typical machine capacities
    machine_loader = GoogleClusterV2DataLoader(base_path, event_type="machine_events", concatenate=True)
    machine_df = next(iter(machine_loader))
    # Get machine capacity (using the mode for robustness)
    # This represents the normalized CPU and memory capacity of a single node.
    cpu_capacity = machine_df['CPU_capacity'].mode()[0]
    mem_capacity = machine_df['memory_capacity'].mode()[0]

    # Load task events to get individual task resource requests
    task_loader = GoogleClusterV2DataLoader(base_path, event_type="task_events", concatenate=True)
    task_df = next(iter(task_loader))
    # Filter to only submitted tasks (event_type=0)
    task_df = task_df[task_df['event_type'] == 0]

    # Calculate total resource requests per job by summing up all task requests for each job
    job_resources = task_df.groupby('job_ID').agg({
        'CPU_request': 'sum',
        'memory_request': 'sum'
    }).reset_index()

    # Calculate nodes required for each job based on CPU and memory requests
    # Using ceiling division to ensure enough nodes are allocated to meet the demand
    job_resources['nodes_required_cpu'] = np.ceil(job_resources['CPU_request'] / cpu_capacity)
    job_resources['nodes_required_mem'] = np.ceil(job_resources['memory_request'] / mem_capacity)
    # The final nodes_required is the maximum of CPU-driven and memory-driven node requirements
    job_resources['nodes_required'] = np.maximum(
        job_resources['nodes_required_cpu'], job_resources['nodes_required_mem']).astype(int)

    # Create a dictionary for quick lookup of nodes_required by job_ID
    nodes_required_map = job_resources.set_index('job_ID')['nodes_required'].to_dict()

    # Filter out jobs with 0 nodes required (i.e., no resource requests)
    num_jobs_before_filter = len(job_resources)
    job_resources = job_resources[job_resources['nodes_required'] > 0]
    num_jobs_after_filter = len(job_resources)
    print(f"Filtered out {num_jobs_before_filter - num_jobs_after_filter} jobs with 0 resource requests.")

    print("Job resource requirements (after filtering):")
    print(job_resources.head())

    # Load submit events
    loader = GoogleClusterV2DataLoader(base_path, event_type="job_events", concatenate=True)
    df = next(iter(loader))
    for col in ("timestamp", "job_ID", "event_type"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
    df = df[df["event_type"] == 0]
    df["timestamp"] = df["timestamp"].astype(float) / 1e6  # convert from microseconds → seconds
    t0 = df["timestamp"].min()
    # t1 = df["timestamp"] - t0  # Unused

    # Get trace quanta
    trace_quanta = config['TRACE_QUANTA']

    # Load task usage
    usage_loader = GoogleClusterV2DataLoader(base_path, event_type="task_usage", concatenate=True)
    usage_df = next(iter(usage_loader))

    # Convert microseconds → seconds for task usage
    usage_df["start_time"] = usage_df["start_time"].astype(float) / 1e6
    usage_df["end_time"] = usage_df["end_time"].astype(float) / 1e6

    # Build per-job start and end times (seconds since trace-start)
    usage_map_start = usage_df.groupby("job_ID")["start_time"].min().to_dict()
    usage_map_end = usage_df.groupby("job_ID")["end_time"].max().to_dict()

    # rename to avg
    if "CPU_usage_rate" in usage_df.columns:
        usage_df.rename(columns={"CPU_usage_rate": "CPU_usage_avg"}, inplace=True)
    usage_df["job_ID"] = usage_df["job_ID"].astype(int)
    usage_df["CPU_usage_avg"] = usage_df["CPU_usage_avg"].astype(float)
    usage_map = usage_df.groupby("job_ID")["CPU_usage_avg"].apply(lambda s: s.to_numpy()).to_dict()

    # print(usage_map)

    # Filter to jobs with usage data AND valid resource requests
    df = df[df["job_ID"].isin(usage_map) & df["job_ID"].isin(job_resources['job_ID'])]

    jobs: List[Any] = []
    jid_f = kwargs.get('jid', '*')
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading jobs"):

        jid = int(row["job_ID"])
        start = usage_map_start[jid] - t0
        end = usage_map_end[jid] - t0
        wall = end - start

        # nodes_required = int(nodes_required_map.get(jid, 1)) # Default to 1 if not found
        # nodes_required = int(nodes_required_map.get(jid))  # Unused

        if jid_f != '*' and str(jid) != str(jid_f):
            continue
        trace = usage_map[jid]
        # ensure gpu_trace is same length as cpu_trace
        gpu_trace = np.zeros_like(trace, dtype=float)

        # nodes_required should be a positive int
        nr = int(nodes_required_map.get(jid, 1))
        if nr < 1:
            nr = 1

        job_d = job_dict(
            nodes_required=nr,
            name=f"job_{jid}",
            account=f"user_{row.get('user_name', 'unknown')}",
            cpu_trace=trace,
            gpu_trace=gpu_trace,
            nrx_trace=[], ntx_trace=[],
            end_state="UNKNOWN", scheduled_nodes=[],
            id=jid, priority=int(row.get('scheduling_class', 0)),
            # submit_time=row["timestamp"], time_limit=0,
            submit_time=start, time_limit=0,
            start_time=start, end_time=end,
            expected_run_time=wall, trace_time=row["timestamp"],
            trace_start_time=start, trace_end_time=end, trace_quanta=trace_quanta
        )
        # Wrap dict in a real Job so telemetry.save_snapshot() can use __dict__
        # if nodes_required > 0:
        jobs.append(Job(job_d))

    # Compute simulation span: start at t=0, end at the latest job finish
    telemetry_start = 0
    telemetry_end = int(max(usage_map_end.values()) - t0)
    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start, telemetry_end=telemetry_end,
        # gcloud dataset timestamps are already relative, and it doesn't list a start exact date.
        start_date=datetime.fromisoformat("2011-05-02T00:00:00Z"),
    )
