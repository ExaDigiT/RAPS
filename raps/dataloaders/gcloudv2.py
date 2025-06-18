import os
import re
from typing import List, Optional, Generator, Tuple, Any, Union

import numpy as np
import pandas as pd

from raps.job import job_dict  # ensure RAPS is in PYTHONPATH

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
    def __init__(self, base_path: str, event_type: str="job_events",
                 file_indices: Optional[List[int]]=None, concatenate: bool=True):
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


def load_data(data_path: Union[str, List[str]], **kwargs: Any) -> Tuple[List[Any], float, float]:
    # Unpack list
    if isinstance(data_path, list):
        if len(data_path)==1:
            data_path=data_path[0]
        else:
            raise ValueError(f"Expected single path, got {data_path}")
    base_path = os.path.expanduser(data_path)

    # Load submit events
    loader = GoogleClusterV2DataLoader(base_path, event_type="job_events", concatenate=True)
    df = next(iter(loader))
    for col in ("timestamp","job_ID","event_type"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
    df = df[df["event_type"]==0]
    df["timestamp"] = df["timestamp"].astype(float)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()

    # Load task usage
    usage_loader = GoogleClusterV2DataLoader(base_path, event_type="task_usage", concatenate=True)
    usage_df = next(iter(usage_loader))
    # rename to avg
    if "CPU_usage_rate" in usage_df.columns:
        usage_df.rename(columns={"CPU_usage_rate":"CPU_usage_avg"}, inplace=True)
    usage_df["job_ID"] = usage_df["job_ID"].astype(int)
    usage_df["CPU_usage_avg"] = usage_df["CPU_usage_avg"].astype(float)
    usage_map = usage_df.groupby("job_ID")["CPU_usage_avg"].apply(lambda s: s.to_numpy()).to_dict()

    # Filter to jobs with usage data
    df = df[df["job_ID"].isin(usage_map)]

    jobs: List[Any] = []
    jid_f = kwargs.get('jid','*')
    for _, row in df.iterrows():
        jid = int(row["job_ID"])
        if jid_f!='*' and str(jid)!=str(jid_f): continue
        trace = usage_map[jid]
        # ensure gpu_trace is same length
        gpu_trace = np.zeros_like(trace)
        jobs.append(job_dict(
            nodes_required=1,
            name=f"job_{jid}",
            account=f"user_{row.get('user_name','unknown')}",
            cpu_trace=trace,
            gpu_trace=gpu_trace,
            nrx_trace=[], ntx_trace=[],
            end_state="UNKNOWN", scheduled_nodes=[],
            id=jid, priority=int(row.get('scheduling_class',0)),
            submit_time=row["timestamp"], time_limit=0,
            start_time=row["timestamp"], end_time=row["timestamp"]+1.0,
            wall_time=1.0, trace_time=row["timestamp"],
            trace_start_time=float(t0), trace_end_time=float(t1)
        ))
    return jobs, 0, 10000
