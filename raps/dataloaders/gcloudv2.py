import pandas as pd
import fsspec
import os
import re 
import math
import numpy as np
from typing import List, Dict, Optional, Union, Generator, Any

# Assuming this script is located in raps/dataloaders/
# Adjust the path if your raps/job.py is located differently
try:
    from ..job import job_dict
except ImportError:
    # Fallback for direct script execution/testing outside RAPS structure
    print("Warning: Could not import 'job_dict' directly. Using a dummy job_dict for testing.")
    class job_dict:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __repr__(self):
            return f"DummyJobDict(id={getattr(self, 'id', 'N/A')})"


class GoogleClusterV2DataLoader:
    """
    A custom dataloader for Google Cluster Traces V2 (2011 dataset),
    designed to read locally downloaded .csv.gz files in an ExaDigiT/RAPS style.

    This dataloader supports loading data from local subdirectories for different
    event types and can handle gzipped CSV files. It allows for selective loading
    of specific event types and file indices.
    """

    # This BASE_LOCAL_PATH will now effectively be managed by the `load_data` function
    # which passes it via the `base_data_path` argument to __init__.
    # It remains here as a default for direct instantiation/testing of the class.
    BASE_LOCAL_PATH = "/Users/w1b/data/gcloud/v2/google_cluster_data_2011_sample/"
    
    SUPPORTED_EVENT_TYPES = [
        "machine_events", "job_events", "task_events", "task_usage",
    ]
    SUPPORTED_FORMATS = ["csv"] 

    V2_COLUMN_NAMES = {
        "job_events": [
            "time", "missing_col_1", "job_ID", "event_type", "user_ID",
            "scheduling_class", "job_name", "logical_job_name",
            "number_of_tasks", "CPU_request", "memory_request"
        ],
        "machine_events": [
            "time", "machine_ID", "event_type", "platform_ID",
            "CPU_capacity", "memory_capacity"
        ],
        "task_events": [
            "time", "missing_col_1", "job_ID", "task_index", "machine_ID",
            "event_type", "user_ID", "scheduling_class", "priority",
            "CPU_request", "memory_request", "disk_space_request", "constraints"
        ],
        "task_usage": [
            "start_time", "end_time", "job_ID", "task_index", "machine_ID",
            "CPU_usage_rate", "memory_usage_avg", "memory_usage_max",
            "disk_IO_time_avg", "disk_IO_time_max", "CPUs_allocated",
            "memory_allocated", "sample_duration", "missing_col_13",
            "missing_col_14", "missing_col_15", "missing_col_16",
            "missing_col_17", "missing_col_18", "missing_col_19" # Up to 20 columns observed
        ]
    }

    def __init__(self,
                 event_types: Optional[Union[str, List[str]]] = None,
                 file_indices: Optional[Union[int, List[int]]] = None,
                 read_options: Optional[Dict[str, Any]] = None,
                 concatenate_files: bool = True,
                 base_data_path: Optional[str] = None): 
        """
        Initializes the GoogleClusterV2DataLoader to read from local V2 trace files.

        Args:
            event_types (Optional[Union[str, List[str]]]):
                Specific event types to load. If None, all supported event types will be considered.
            file_indices (Optional[Union[int, List[int]]]):
                Specific numerical indices of parts to load. If None, all available files for the selected types.
            read_options (Optional[Dict[str, Any]]):
                Additional options passed directly to pandas.read_csv().
            concatenate_files (bool):
                If True, all loaded files will be concatenated into a single pandas DataFrame.
                If False, `__iter__` will yield individual DataFrames.
            base_data_path (Optional[str]): The base path to the local data directory.
                                            This is the root that contains subdirectories like 'job_events'.
        """
        self.event_types = [event_types] if isinstance(event_types, str) else event_types
        self.file_indices = [file_indices] if isinstance(file_indices, int) else file_indices
        self.concatenate_files = concatenate_files

        # Set default read options specific to V2 CSVs
        self.read_options = read_options.copy() if read_options is not None else {}
        if 'header' not in self.read_options:
            self.read_options['header'] = None # V2 CSVs do not have a header row
        if 'dtype' not in self.read_options:
            self.read_options['dtype'] = {}
        self.read_options['dtype']['time'] = 'int64' # Force 'time' to be read as integer
        self.read_options['dtype']['start_time'] = 'int64' # For task_usage
        self.read_options['dtype']['end_time'] = 'int64'   # For task_usage


        # The effective base path for this DataLoader instance will be where the event_type_dirs are.
        # This is the key path that load_data will correctly provide.
        self._current_base_path = base_data_path if base_data_path is not None else self.BASE_LOCAL_PATH

        self._fs = fsspec.AbstractFileSystem() 
        self._all_file_paths = [] 
        self._discover_files()

    def _discover_files(self):
        """
        Discovers local V2 trace files based on specified event types and indices.
        Populates self._all_file_paths with absolute file paths.
        """
        event_types_to_consider = self.event_types if self.event_types else self.SUPPORTED_EVENT_TYPES

        self._all_file_paths = [] 

        for event_type in event_types_to_consider:
            event_type_dir = os.path.join(self._current_base_path, event_type)

            if event_type in self.V2_COLUMN_NAMES:
                # Add names to read_options for this specific type loading instance
                self.read_options['names'] = self.V2_COLUMN_NAMES[event_type]
            else:
                self.read_options.pop('names', None) # Remove names if not defined for this type
                print(f"Warning: No explicit column names defined for '{event_type}'. Pandas will infer names.")
            
            if not os.path.isdir(event_type_dir):
                print(f"Warning: Local directory for '{event_type}' not found: '{event_type_dir}'. Skipping this type.")
                continue

            if self.file_indices:
                for idx in self.file_indices:
                    filename_pattern_re = rf"part-{idx:05d}-of-\d{{5}}\.csv\.gz"
                    
                    found_indexed_file = False
                    for filename in os.listdir(event_type_dir):
                        if re.fullmatch(filename_pattern_re, filename):
                            self._all_file_paths.append(os.path.join(event_type_dir, filename))
                            found_indexed_file = True
                            break 
                    
                    if not found_indexed_file:
                        print(f"Warning: Specific file '{event_type}/part-{idx:05d}-of-*.csv.gz' not found in '{event_type_dir}'.")
            else:
                for filename in os.listdir(event_type_dir):
                    if filename.startswith("part-") and filename.endswith(".csv.gz"):
                        self._all_file_paths.append(os.path.join(event_type_dir, filename))

        self._all_file_paths = sorted(list(set(self._all_file_paths)))
        
        if not self._all_file_paths:
            print(f"Warning: No local V2 trace files found in '{self._current_base_path}' matching the criteria.")

    def __len__(self) -> int:
        return len(self._all_file_paths)

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        if not self._all_file_paths:
            return 

        all_data_frames = []
        total_files = len(self._all_file_paths)
        
        print(f"\nStarting to load {total_files} selected V2 trace files from '{self._current_base_path}'...")

        for i, file_path in enumerate(self._all_file_paths):
            file_name = os.path.basename(file_path)
            
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            print(f"[{i + 1}/{total_files}] Loading '{file_name}' ({file_size_mb:.2f} MB)...", end='', flush=True)
            
            df = None
            try:
                df = pd.read_csv(file_path, compression='gzip', **self.read_options)
                print(f" -> OK. Shape: {df.shape}")
            except Exception as e:
                print(f" -> FAILED. Error: {e}")
                print(f"  Failed to read CSV file '{file_name}'. Double-check CSV format (e.g., separator, header) or file integrity.")
                continue 

            if df is not None:
                if self.concatenate_files:
                    all_data_frames.append(df)
                else:
                    yield df 

        if self.concatenate_files and all_data_frames:
            final_df = pd.concat(all_data_frames, ignore_index=True)
            print(f"\nAll selected V2 files concatenated. Final DataFrame shape: {final_df.shape}")
            yield final_df
        elif self.concatenate_files and not all_data_frames:
            print("\nNo DataFrames were loaded to concatenate from the selected V2 files.")

    def get_data_for_type(self, event_type: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        A convenience method to load data for a single event type from the V2 dataset,
        up to a specified number of files. (Format is fixed to CSV for V2).
        """
        if event_type not in self.SUPPORTED_EVENT_TYPES:
            raise ValueError(f"Unsupported event type: '{event_type}'. Choose from {self.SUPPORTED_EVENT_TYPES}")

        original_event_types = self.event_types
        original_file_indices = self.file_indices
        original_concatenate_files = self.concatenate_files
        original_current_base_path = self._current_base_path 

        self.event_types = [event_type]
        self.concatenate_files = True 

        temp_file_indices = None
        if limit is not None:
            temp_file_indices = list(range(limit))
        self.file_indices = temp_file_indices

        self._discover_files() 

        combined_df = pd.DataFrame()
        for df_chunk in self: 
            combined_df = df_chunk 

        self.event_types = original_event_types
        self.file_indices = original_file_indices
        self.concatenate_files = original_concatenate_files
        self._current_base_path = original_current_base_path 
        self._discover_files() 

        return combined_df

# --- MANDATORY RAPS `load_data` FUNCTION ---
# This function is the entry point that RAPS's main.py will call.
def load_data(
    data_path: Union[str, List[str]], **kwargs) -> tuple[List[Any], float, float]: # RAPS expects a list of job_dict instances, start_time, end_time
    """
    RAPS data loading entry point for Google Cluster Trace V2 (2011) data.

    Loads data from a specified local path, assuming it contains subdirectories
    like 'job_events', 'task_events', etc., filled with .csv.gz files.
    It returns a list of RAPS job_dict instances, along with the global start
    and end timestamps of the loaded data.

    Args:
        data_path (Union[str, List[str]]): The base path to the local V2 data directory.
                         Expected to be the directory that *contains* the
                         'google_cluster_data_2011_sample' subdirectory.
                         Can be a string or a list containing a single string.
                         Example: `~/data/gcloud/v2/`

    Returns:
        tuple[List[Any], float, float]:
            - A list of RAPS `job_dict` instances.
            - The global minimum timestamp (float) found across all loaded data.
            - The global maximum timestamp (float) found across all loaded data.
            Returns ([], 0.0, 0.0) if no data or time info found.
    """
    
    # --- FIX 1: Handle data_path potentially being a list (from argparse) ---
    if isinstance(data_path, list):
        if len(data_path) == 1:
            data_path_str = data_path[0]
        else:
            raise ValueError(
                f"load_data expected a single base data path, but received a list of multiple paths: {data_path}. "
                f"Please ensure RAPS passes a single path."
            )
    else:
        data_path_str = data_path 
    # --- END FIX 1 ---

    # Expand the user home directory if '~' is used in data_path_str
    expanded_data_path = os.path.expanduser(data_path_str)
    # Ensure it ends with a slash for consistency with os.path.join later
    if not expanded_data_path.endswith(os.sep):
        expanded_data_path += os.sep

    # This dictionary will store DataFrames for all event types loaded by this function
    loaded_dfs: Dict[str, pd.DataFrame] = {}
    
    # Load all supported event types (job_events, task_events, etc.)
    # We set event_types=None and file_indices=None to load all available files for each type
    # from the automatically detected subdirectories.
    dataloader = GoogleClusterV2DataLoader(
        event_types=None,        # Load all supported types
        file_indices=None,       # Load all files found for each type
        read_options=None,       # Use default read_options defined in DataLoader
        concatenate_files=True,  # Get one concatenated DF per type
        base_data_path=expanded_data_path # This is the RAPS-provided path to the directory *above* the data folder
    )
    
    # Initialize global min/max timestamps for the entire dataset
    global_min_time = float(math.inf)
    global_max_time = float(-math.inf)

    # Loop through the dataloader to get all concatenated DataFrames for each event type
    for event_type_key in dataloader.SUPPORTED_EVENT_TYPES: # Iterate through explicitly supported types
        # Create a temporary DataLoader instance just to load this specific event type
        # from the correct subpath within expanded_data_path
        temp_dataloader_for_type = GoogleClusterV2DataLoader(
            event_types=event_type_key,
            file_indices=None, # Load all files for this specific type
            read_options=None, # Use default read_options
            concatenate_files=True,
            base_data_path=expanded_data_path # Pass the RAPS base path
        )
        
        # This loop will run once for each event type, yielding its concatenated DataFrame
        for df_current_type in temp_dataloader_for_type:
            if not df_current_type.empty:
                loaded_dfs[event_type_key] = df_current_type
                print(f"RAPS: Successfully loaded '{event_type_key}'. DataFrame shape: {df_current_type.shape}")

                # Update global min/max times if a 'time' column exists
                if 'time' in df_current_type.columns:
                    current_min = df_current_type['time'].min()
                    current_max = df_current_type['time'].max()
                    if current_min < global_min_time:
                        global_min_time = current_min
                    if current_max > global_max_time:
                        global_max_time = current_max
            else:
                print(f"RAPS: No data loaded for event type '{event_type_key}'.")
                
    print("\n--- RAPS: Data loading complete for individual types ---")

    # --- FIX 2: Select and prepare the primary 'jobs' list for RAPS ---
    # RAPS main.py is iterating over `jobs`, expecting `job['wall_time']` and `job['start_time']`.
    # This means `jobs` must be a list of dictionaries (or job_dict instances).
    jobs_list_for_rap: List[Any] = [] 

    # Prioritize task_events for primary job records, otherwise use job_events.
    raw_primary_records_df = pd.DataFrame() 
    if 'task_events' in loaded_dfs and not loaded_dfs['task_events'].empty:
        raw_primary_records_df = loaded_dfs['task_events'].copy()
        print(f"RAPS: Selected 'task_events' as the primary source for job records.")
    elif 'job_events' in loaded_dfs and not loaded_dfs['job_events'].empty:
        raw_primary_records_df = loaded_dfs['job_events'].copy()
        print(f"RAPS: Selected 'job_events' as the primary source for job records (task_events not available/empty).")
    else:
        print("RAPS: Warning: Neither 'task_events' nor 'job_events' found/loaded for primary 'jobs' data. Cannot create job records.")
        # Return empty list and 0.0 times if no primary data
        return [], 0.0, 0.0

    if not raw_primary_records_df.empty:
        # --- FIX 3: Prepare raw_primary_records_df with RAPS-expected columns ---
        # Map V2 'time' column to RAPS 'submit_time' and 'start_time'
        if 'time' in raw_primary_records_df.columns:
            raw_primary_records_df['submit_time'] = raw_primary_records_df['time']
            raw_primary_records_df['start_time'] = raw_primary_records_df['time']
        else:
            raw_primary_records_df['submit_time'] = 0 
            raw_primary_records_df['start_time'] = 0 
            print("Warning: 'time' column not found in primary records DataFrame. Using 0 for submit/start_time.")

        # Add 'wall_time'. V2 trace does not have explicit wall_time per job/task.
        # This is a dummy value for RAPS's internal calculations.
        raw_primary_records_df['wall_time'] = 1 # Dummy: 1 microsecond duration

        # Add 'end_time' to the DataFrame for internal consistency if needed later
        # (though RAPS main.py calculates it, having it can be useful)
        raw_primary_records_df['end_time'] = raw_primary_records_df['start_time'] + raw_primary_records_df['wall_time']

        # --- FIX 4: Create job_dict instances and populate jobs_list_for_rap ---
        # Get the jid (job ID filter) from kwargs, defaulting to '*'
        jid_filter = kwargs.get('jid', '*')

        # Filter to 'submit' events to represent distinct job creations
        submit_records_df = raw_primary_records_df[
            raw_primary_records_df.get('event_type') == 0 # Event type 0 is 'submit'
        ].copy() if 'event_type' in raw_primary_records_df.columns else raw_primary_records_df.copy()

        if 'job_ID' not in submit_records_df.columns: 
             submit_records_df['job_ID'] = submit_records_df['task_index'] if 'task_index' in submit_records_df.columns else range(len(submit_records_df))
             print("Warning: 'job_ID' not found. Using 'task_index' or row index for job_id.")
        
        # Make job_ID unique in case 'task_index' was used and job_ID wasn't.
        # This ensures unique RAPS job_dict IDs.
        submit_records_df['unique_job_id'] = submit_records_df['job_ID'].astype(str) + "_" + submit_records_df['start_time'].astype(str)

        for index, row in submit_records_df.iterrows():
            job_id_from_trace = row['job_ID'] # The original job_ID from the trace
            
            # Apply RAPS's jid filter (from main.py example)
            if jid_filter != '*' and str(job_id_from_trace) != str(jid_filter): 
                continue 

            # --- Map V2 Data to job_dict arguments ---
            nodes_required = 1 # Dummy: V2 doesn't specify nodes_required directly per job event
            name = f"job_{job_id_from_trace}"
            account = f"user_{row['user_ID']}" if 'user_ID' in row else "unknown_user"
            priority = row['priority'] if 'priority' in row else 0

            # Trace data arrays are empty as per V2 characteristics
            cpu_trace = np.array([]) 
            gpu_trace = np.array([]) # V2 has no GPUs
            nrx_trace = np.array([]) 
            ntx_trace = np.array([]) 

            end_state = "UNKNOWN" # Final job state requires complex aggregation of task events
            scheduled_nodes = [] # Requires scheduling logic, not directly in raw event
            
            # Global trace times (already calculated above)
            trace_start_time = float(global_min_time) if global_min_time != float(math.inf) else 0.0
            trace_end_time = float(global_max_time) if global_max_time != float(-math.inf) else 0.0
            
            # This specific record's time (from its 'time' column)
            trace_time_for_record = row['time'] if 'time' in row else 0 

            job_info = job_dict(
                nodes_required=nodes_required,
                name=name,
                account=account,
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                nrx_trace=nrx_trace,
                ntx_trace=ntx_trace,
                end_state=end_state,
                scheduled_nodes=scheduled_nodes,
                id=job_id_from_trace, # Use the original job ID from the trace
                priority=priority,
                submit_time=row['submit_time'],
                time_limit=0, # V2 doesn't have explicit time_limit per job_event
                start_time=row['start_time'], # RAPS uses this for simulation start
                end_time=row['end_time'],     # RAPS uses this for simulation end
                wall_time=row['wall_time'],   # RAPS uses this for duration
                trace_time=trace_time_for_record, # The timestamp of *this specific event* record
                trace_start_time=trace_start_time, # Global trace start time
                trace_end_time=trace_end_time      # Global trace end time
            )
            jobs_list_for_rap.append(job_info)
        print(f"RAPS: Created {len(jobs_list_for_rap)} job_dict instances.")
    else:
        print("RAPS: No primary records DataFrame available to create job_dict instances.")

    # Convert global min/max times to float
    final_timestep_start = float(global_min_time) if global_min_time != float(math.inf) else 0.0
    final_timestep_end = float(global_max_time) if global_max_time != float(-math.inf) else 0.0

    print(f"RAPS: Final global time range determined: Start={final_timestep_start}, End={final_timestep_end}")
    
    # Return the three values RAPS expects:
    # (list of job_dict instances, global min time, global max time)
    return jobs_list_for_rap, final_timestep_start, final_timestep_end


# --- Example Usage (for direct script execution/testing the load_data function) ---
if __name__ == "__main__":
    # IMPORTANT: Adjust this path to match your local setup precisely.
    # This path should be the DIRECTORY THAT RAPS's `-f` ARGUMENT POINTS TO.
    # e.g., if you run `main.py -f /Users/w1b/data/gcloud/v2`, then this variable is '/Users/w1b/data/gcloud/v2/'.
    # And inside THAT directory, you should have `google_cluster_data_2011_sample/`
    RAPS_SIMULATED_BASE_DIR = "/Users/w1b/data/gcloud/v2/" 
    
    print("--- Running direct tests of the load_data function ---")

    print("\n--- Test 1: Loading all event types (default behavior for a RAPS integration) ---")
    jobs_list_test1, start_time_test1, end_time_test1 = load_data([RAPS_SIMULATED_BASE_DIR], system="dummy_system_name") 
    
    if jobs_list_test1: 
        print(f"\nSummary of Test 1 (Primary jobs list loaded):")
        print(f"- Number of job/task records: {len(jobs_list_test1)}")
        if jobs_list_test1 and hasattr(jobs_list_test1[0], 'start_time') and hasattr(jobs_list_test1[0], 'wall_time'):
            print(f"- First record (id={jobs_list_test1[0].id}): submit_time={jobs_list_test1[0].submit_time}, start_time={jobs_list_test1[0].start_time}, wall_time={jobs_list_test1[0].wall_time}")
            # print(f"- Full first record details: {jobs_list_test1[0].__dict__}") 
        print(f"- Global Start time: {start_time_test1}, Global End time: {end_time_test1}")
    else:
        print("\nTest 1: No primary jobs list loaded. Check specified paths and downloaded files.")

    print("\n--- Test 2: Loading specific event types and file indices ---")
    jobs_list_test2, start_time_test2, end_time_test2 = load_data(
        [RAPS_SIMULATED_BASE_DIR], 
        event_types=["job_events"], 
        file_indices=[0], 
        read_options={'header': 0}, 
        another_rap_param=123 
    )

    if jobs_list_test2: 
        print(f"\nSummary of Test 2 (Primary jobs list loaded):")
        print(f"- Number of job/task records: {len(jobs_list_test2)}")
        if jobs_list_test2 and hasattr(jobs_list_test2[0], 'start_time') and hasattr(jobs_list_test2[0], 'wall_time'):
            print(f"- First record (id={jobs_list_test2[0].id}): submit_time={jobs_list_test2[0].submit_time}, start_time={jobs_list_test2[0].start_time}, wall_time={jobs_list_test2[0].wall_time}")
            # print(f"- Full first record details: {jobs_list_test2[0].__dict__}") 
        print(f"- Global Start time: {start_time_test2}, Global End time: {end_time_test2}")
    else:
        print("\nTest 2: No primary jobs list loaded. Check path, types, and indices.")

    print("\n--- RAPS Dataloader (V2) script demonstration complete ---")
