#!/bin/bash

# Define the base GCS path for the 2011 (V2) dataset
GCS_BASE_PATH="gs://clusterdata-2011-2"
LOCAL_DIR="./google_cluster_data_2011_sample" # Local directory to save files
NUM_FILES_PER_TYPE=1 # Adjust this number: 1 is very small, 5-10 is a decent sample

# Create the local base directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Define the event types present in the 2011 (V2) dataset
EVENT_TYPES=(
    "machine_events"
    "job_events"
    "task_events"
    "task_usage"
)

echo "Starting download of Google Cluster Data V2 (2011) sample..."

# Loop through each event type
for event_type in "${EVENT_TYPES[@]}"; do
    echo "Processing event type: ${event_type}"

    # Create a local subdirectory for each event type
    mkdir -p "${LOCAL_DIR}/${event_type}"

    # List files in the current event type's GCS directory, take the first N, and download them
    # Added '2>/dev/null' to suppress BrokenPipeError messages from gsutil ls
    gsutil ls "${GCS_BASE_PATH}/${event_type}/part-*.csv.gz" 2>/dev/null | head -n "${NUM_FILES_PER_TYPE}" | while read -r gcs_path; do
        echo "  Downloading $(basename "$gcs_path")..."
        gsutil cp "$gcs_path" "${LOCAL_DIR}/${event_type}/"
    done
done

echo "---"
echo "Download complete. Files are in: $LOCAL_DIR"
echo "You've downloaded a sample of the 2011 (V2) Google Cluster Traces."
