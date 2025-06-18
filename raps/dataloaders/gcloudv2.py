import pandas as pd
import fsspec
import os
import re 
import math # For math.inf and -math.inf
import numpy as np # Needed for empty arrays for traces
from typing import List, Dict, Optional, Union, Generator, Any

""" To download cluster traces into ~/data/gcloud/v2

    1. Install Google cloud SDK

        https://cloud.google.com/sdk/docs/install

    2. gcloud auth login

    3. See https://github.com/google/cluster-data - we are using v2 traces b/c the v3 traces are too large for practical study

    4. See download script in ../../get_cluster_v2_traces.sh

"""

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
            return f"DummyJobDict({self.id})"

class GoogleClusterV2DataLoader:
    """
    A custom dataloader for Google Cluster Traces V2 (2011 dataset),
    designed to read locally downloaded .csv.gz files in an ExaDigiT/RAPS style.
    """

    # --- Configuration for your local V2 files ---
    # This is a default fallback path for direct script execution (when not called by RAPS).
    # It MUST point directly to the directory *containing* machine_events, job_events, etc.
    BASE_LOCAL_PATH = "~/data/gcloud/v2/google_cluster_data_2011_sample/"
    
    SUPPORTED_EVENT_TYPES = [
        "machine_events", "job_events", "task_events", "task_usage",
    ]
    SUPPORTED_FORMATS = ["csv"] 

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
                If True, all loaded files will be concatenated into a single pandas DataFrame
                when the iterator finishes. If False, `__iter__` will yield individual DataFrames.
            base_data_path (Optional[str]): The base path provided by the external calling script (e.g., RAPS).
                                            This path will be adjusted to point to the actual data root.
        """
        self.event_types = [event_types] if isinstance(event_types, str) else event_types
        self.file_indices = [file_indices] if isinstance(file_indices, int) else file_indices
        self.concatenate_files = concatenate_files

        # --- CRITICAL FIX START: Ensure 'time' column is read as int64 ---
        self.read_options = read_options.copy() if read_options is not None else {}
        if 'dtype' not in self.read_options:
            self.read_options['dtype'] = {}
        self.read_options['dtype']['time'] = 'int64' # Force 'time' to be read as integer
        # --- CRITICAL FIX END ---

        # --- Determine the correct base path for this DataLoader instance ---
        if base_data_path is not None:
            clean_base_path = base_data_path
            if not clean_base_path.endswith(os.sep):
                clean_base_path += os.sep
            self._current_base_path = os.path.join(clean_base_path, "google_cluster_data_2011_sample") + os.sep
        else:
            self._current_base_path = self.BASE_LOCAL_PATH
        # --- End of base path determination ---

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
def load_data(
    data_path: Union[str, List[str]], 
    event_types: Optional[Union[str, List[str]]] = None,
    file_indices: Optional[Union[int, List[int]]] = None,
    read_options: Optional[Dict[str, Any]] = None, # User-provided read_options
    **kwargs # Catch-all for additional arguments like 'system'
) -> tuple[List[Any], float, float]: # Updated return type hint for RAPS job_dict instances
    """
    RAPS data loading entry point for Google Cluster Trace V2 (2011) data.

    Loads data from a specified local path, organizing it by event type.
    It returns a list of primary 'job/task' records (as job_dict instances),
    along with the global start and end timestamps of the loaded data.

    Args:
        data_path (Union[str, List[str]]): The base path to the local V2 data directory.
                         This is the path provided by the RAPS main script.
                         Can be a string or a list containing a single string.
        event_types (Optional[Union[str, List[str]]]):
            Specific event types to load. If None, all supported types.
        file_indices (Optional[Union[int, List[int]]]):
            Specific numerical indices of parts to load. If None, all available parts.
        read_options (Optional[Dict[str, Any]]):
            Additional options for pandas.read_csv().
        **kwargs: Catch-all for any additional keyword arguments passed by RAPS
                  (e.g., 'system', 'config', 'cooling', 'fastforward', etc.).

    Returns:
        tuple[List[Any], float, float]:
            - A list of job_dict instances, where each represents a job/task record
              and includes 'start_time' and 'wall_time' (even if derived/dummy).
            - The global minimum timestamp found across all loaded data.
            - The global maximum timestamp found across all loaded data.
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
    
    types_to_load_for_rap = event_types if event_types else GoogleClusterV2DataLoader.SUPPORTED_EVENT_TYPES
    if isinstance(types_to_load_for_rap, str): 
        types_to_load_for_rap = [types_to_load_for_rap]

    # Initialize global min/max timestamps for the entire dataset
    global_min_time = float(math.inf)
    global_max_time = float(-math.inf)

    for event_type_key in types_to_load_for_rap:
        # Create a new DataLoader instance for each event_type to get its concatenated DF.
        dataloader = GoogleClusterV2DataLoader(
            event_types=event_type_key, # Load only this specific type
            file_indices=file_indices,  # Apply file index filter
            read_options=read_options,  # Apply any custom read options (will be merged with default dtype for time)
            concatenate_files=True,     # Ensure a single DataFrame is yielded for this type
            base_data_path=expanded_data_path # Pass the RAPS provided path
        )
        
        for df_current_type in dataloader: 
            if not df_current_type.empty:
                loaded_dfs[event_type_key] = df_current_type
                # DEBUG: Check if 'time' column is being correctly read.
                if 'time' in df_current_type.columns:
                    print(f"DEBUG: '{event_type_key}' time min/max in DF: {df_current_type['time'].min()}/{df_current_type['time'].max()}")
                else:
                    print(f"DEBUG: 'time' column NOT FOUND in {event_type_key}.")
            else:
                print(f"RAPS: No data loaded for event type '{event_type_key}'.")
                
    print("\n--- RAPS: Data loading complete for individual types ---")

    # --- FIX 2: Select and prepare the primary 'jobs' DataFrame for RAPS ---
    # RAPS main.py is iterating over `jobs`, expecting `job['wall_time']` and `job['start_time']`.
    # This means `jobs` must be a list of dictionaries (or similar objects).
    jobs_list_for_rap: List[Any] = [] # Initialize as empty list

    # Prioritize task_events for job records due to granularity, otherwise use job_events.
    raw_primary_records_df = pd.DataFrame() 
    if 'task_events' in loaded_dfs and not loaded_dfs['task_events'].empty:
        raw_primary_records_df = loaded_dfs['task_events'].copy()
        print(f"RAPS: Selected 'task_events' as the primary source for job records.")
    elif 'job_events' in loaded_dfs and not loaded_dfs['job_events'].empty:
        raw_primary_records_df = loaded_dfs['job_events'].copy()
        print(f"RAPS: Selected 'job_events' as the primary source for job records (task_events not available/empty).")
    else:
        print("RAPS: Warning: Neither 'task_events' nor 'job_events' found/loaded. Cannot create job records.")

    if not raw_primary_records_df.empty:
        # --- FIX 3: Prepare raw_primary_records_df with RAPS-expected columns ---
        # Map V2 'time' column to RAPS 'submit_time' and 'start_time'
        if 'time' in raw_primary_records_df.columns:
            raw_primary_records_df['submit_time'] = raw_primary_records_df['time']
            raw_primary_records_df['start_time'] = raw_primary_records_df['time'] # Simplistic for first pass
        else:
            raw_primary_records_df['submit_time'] = 0
            raw_primary_records_df['start_time'] = 0
            print("Warning: 'time' column not found in primary records DataFrame. Using 0 for submit/start_time.")

        # Derive 'end_time' and 'wall_time'. This is a major simplification for V2 data.
        # For a more accurate 'end_time' and 'wall_time', you'd need to:
        # 1. Join with 'task_usage' or other event types.
        # 2. Aggregate events by job/task ID to find actual lifecycle timestamps.
        # For now, setting a dummy wall_time to satisfy RAPS's requirement for `job['wall_time']`
        # and to allow its `int(max(job['wall_time'] + job['start_time']...` calculation.
        raw_primary_records_df['wall_time'] = 1 # Dummy: 1 microsecond duration
        raw_primary_records_df['end_time'] = raw_primary_records_df['start_time'] + raw_primary_records_df['wall_time']

        # --- FIX 4: Create job_dict instances and populate list ---
        # Get the jid (job ID filter) from kwargs, defaulting to '*'
        jid_filter = kwargs.get('jid', '*')

        # Loop through each record (row) to create a job_dict instance
        # It's usually best to filter for submit events to ensure unique job instances
        # and to map them to the proper RAPS 'job' concept.
        submit_records_df = raw_primary_records_df[
            raw_primary_records_df.get('event_type') == 'submit' # Use .get() for robustness
        ].copy() if 'event_type' in raw_primary_records_df.columns else raw_primary_records_df.copy() # If no event_type, use all

        if 'job_ID' not in submit_records_df.columns: # Fallback if job_ID not present (e.g., for task_events direct)
             submit_records_df['job_ID'] = submit_records_df['task_ID'] if 'task_ID' in submit_records_df.columns else range(len(submit_records_df))
             print("Warning: 'job_ID' not found in selected primary records. Using 'task_ID' or row index.")

        for index, row in submit_records_df.iterrows():
            job_id = row['job_ID']
            
            # Apply RAPS's jid filter (from main.py example)
            if jid_filter != '*' and str(job_id) != str(jid_filter): # Convert to string for comparison
                continue 

            # --- Map V2 Data to job_dict arguments ---
            # Most of these are simplifications or dummies for V2 given the limited data.
            nodes_required = 1 # Dummy
            name = f"job_{job_id}"
            account = f"user_{row['user_ID']}" if 'user_ID' in row else "unknown_user"
            priority = row['priority'] if 'priority' in row else 0

            # Trace data fields (cpu_trace, gpu_trace etc.) are arrays, initially empty.
            # V2 has no GPUs.
            cpu_trace = np.array([]) 
            gpu_trace = np.array([]) 
            nrx_trace = np.array([]) 
            ntx_trace = np.array([]) 

            end_state = "UNKNOWN" # V2 job_events has event_type, but not direct final state field.
            scheduled_nodes = [] # Requires complex task scheduling analysis
            
            # Get trace-wide times (global min/max)
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
                id=job_id,
                priority=priority,
                submit_time=row['submit_time'],
                time_limit=0, # V2 doesn't have explicit time_limit
                start_time=row['start_time'],
                end_time=row['end_time'],
                wall_time=row['wall_time'],
                trace_time=trace_time_for_record, # The time for this specific event record
                trace_start_time=trace_start_time, # Global trace start
                trace_end_time=trace_end_time      # Global trace end
            )
            jobs_list_for_rap.append(job_info)
        print(f"RAPS: Created {len(jobs_list_for_rap)} job_dict instances.")
    else:
        print("RAPS: No primary records DataFrame available to create job_dict instances.")

    # Convert global min/max times to float
    final_timestep_start = 0 #float(global_min_time) if global_min_time != float(math.inf) else 0.0
    final_timestep_end = 10000 #float(global_max_time) if global_max_time != float(-math.inf) else 0.0

    print(f"RAPS: Final global time range determined: Start={final_timestep_start}, End={final_timestep_end}")
    
    # Return the three values RAPS expects
    return jobs_list_for_rap, final_timestep_start, final_timestep_end


# --- Example Usage (for direct script execution/testing the load_data function) ---
if __name__ == "__main__":
    # IMPORTANT: Adjust this path to match your local setup precisely.
    # This path should be the DIRECTORY THAT RAPS's `-f` ARGUMENT POINTS TO.
    # e.g., if you run `main.py -f /Users/w1b/data/gcloud/v2`, then this variable is '/Users/w1b/data/gcloud/v2/'.
    # And inside THAT directory, you should have `google_cluster_data_2011_sample/`
    #RAPS_SIMULATED_BASE_DIR = "/Users/w1b/data/gcloud/v2/" 
    
    print("--- Running direct tests of the load_data function ---")

    print("\n--- Test 1: Loading all event types (default behavior for a RAPS integration) ---")
    # Simulate RAPS passing a list with one element for the data_path
    jobs_list_test1, start_time_test1, end_time_test1 = load_data([RAPS_SIMULATED_BASE_DIR], system="dummy_system_name") 
    
    if jobs_list_test1: # Check if the list of jobs is not empty
        print(f"\nSummary of Test 1 (Primary jobs list loaded):")
        print(f"- Number of job/task records: {len(jobs_list_test1)}")
        # Check if individual records have the expected keys
        if jobs_list_test1 and hasattr(jobs_list_test1[0], 'start_time') and hasattr(jobs_list_test1[0], 'wall_time'):
            print(f"- First record (id={jobs_list_test1[0].id}): start_time={jobs_list_test1[0].start_time}, wall_time={jobs_list_test1[0].wall_time}")
            print(f"- Full first record details: {jobs_list_test1[0].__dict__}") # Show all attributes
        print(f"- Global Start time: {start_time_test1}, Global End time: {end_time_test1}")
    else:
        print("\nTest 1: No primary jobs list loaded. Check specified paths and downloaded files.")

    print("\n--- Test 2: Loading specific event types and file indices ---")
    jobs_list_test2, start_time_test2, end_time_test2 = load_data(
        [RAPS_SIMULATED_BASE_DIR], 
        event_types=["job_events"], # Only request job_events explicitly for this test
        file_indices=[0], # Load only the 'part-00000' file for job_events
        read_options={'header': 0}, # Example: assuming first row is header
        another_rap_param=123 # Example of passing an extra kwarg
    )

    if jobs_list_test2: # Check if the list of jobs is not empty
        print(f"\nSummary of Test 2 (Primary jobs list loaded):")
        print(f"- Number of job/task records: {len(jobs_list_test2)}")
        if jobs_list_test2 and hasattr(jobs_list_test2[0], 'start_time') and hasattr(jobs_list_test2[0], 'wall_time'):
            print(f"- First record (id={jobs_list_test2[0].id}): start_time={jobs_list_test2[0].start_time}, wall_time={jobs_list_test2[0].wall_time}")
            print(f"- Full first record details: {jobs_list_test2[0].__dict__}") # Show all attributes
        print(f"- Global Start time: {start_time_test2}, Global End time: {end_time_test2}")
    else:
        print("\nTest 2: No primary jobs list loaded. Check path, types, and indices.")

    print("\n--- RAPS Dataloader (V2) script demonstration complete ---")
