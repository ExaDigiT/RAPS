import os
import pandas as pd
import csv
from tqdm import tqdm

def generate_local_metadata(local_dataset_root_path):
    mit_dir = os.path.dirname(os.path.abspath(__file__))
    source_data_dir = os.path.join(mit_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    print(f"Generating metadata in: {source_data_dir}")

    # --- Generate file_list.csv ---
    file_list_path = os.path.join(source_data_dir, 'file_list.csv')
    print(f"Creating {file_list_path}...")
    all_files = []
    for root, _, files in os.walk(local_dataset_root_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    with open(file_list_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='	')
        writer.writerow(["File Name", "Size (MB)"])
        for full_path in tqdm(all_files, desc="Generating file_list.csv"):
            relative_path = os.path.relpath(full_path, local_dataset_root_path)
            file_size_bytes = os.path.getsize(full_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            writer.writerow([relative_path, f"{file_size_mb:.2f}"])
    print(f"Finished creating {file_list_path}")

    # --- Generate job_user_date.csv ---
    job_user_date_path = os.path.join(source_data_dir, 'job_user_date.csv')
    print(f"Creating {job_user_date_path} (resumable)...")

    all_summary_files = []
    for root, _, files in os.walk(local_dataset_root_path):
        for file in files:
            if file.endswith('-summary.csv'):
                all_summary_files.append(os.path.join(root, file))

    processed_job_ids = set()
    if os.path.exists(job_user_date_path):
        try:
            existing_df = pd.read_csv(job_user_date_path)
            processed_job_ids = set(existing_df['job_id'].tolist())
            write_mode = 'a'
            header = False
        except pd.errors.EmptyDataError:
            write_mode = 'w'
            header = True
    else:
        write_mode = 'w'
        header = True

    with open(job_user_date_path, write_mode, newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["job_id", "filename", "start", "end", "node_count"])

        for full_summary_path in tqdm(all_summary_files, desc="Generating job_user_date.csv"):
            file = os.path.basename(full_summary_path)
            jobid = int(file.split('-')[0])

            if jobid in processed_job_ids:
                continue # Skip already processed

            try:
                df = pd.read_csv(full_summary_path)
                st_time = df['Min_EpochTime'].min()
                ed_time = df['Max_EpochTime'].max()
                node_count = df.Node.unique().shape[0]
                relative_filename = os.path.relpath(full_summary_path, local_dataset_root_path)
                writer.writerow([jobid, relative_filename, st_time, ed_time, node_count])
                processed_job_ids.add(jobid)
            except Exception as e:
                print(f"Error processing local summary file {full_summary_path}: {e}")
    print(f"Finished creating {job_user_date_path}")

    # --- Generate job_user_date_full.csv ---
    job_user_date_full_path = os.path.join(source_data_dir, 'job_user_date_full.csv')
    
    # Search for slurm-log.csv anywhere within the local dataset root
    slurm_log_path = None
    for root, _, files in os.walk(local_dataset_root_path):
        if 'slurm-log.csv' in files:
            slurm_log_path = os.path.join(root, 'slurm-log.csv')
            break

    if slurm_log_path is None:
        print(f"Warning: slurm-log.csv not found in {local_dataset_root_path}. Skipping job_user_date_full.csv generation.")
        return

    if os.path.exists(job_user_date_path) and os.path.exists(slurm_log_path):
        print(f"Creating {job_user_date_full_path}...")
        try:
            job_index_df = pd.read_csv(job_user_date_path)
            slurm_df = pd.read_csv(slurm_log_path)
            slurm_df = slurm_df[['id_job', 'id_user']]
            final_df = job_index_df.merge(slurm_df, left_on='job_id', right_on='id_job', how='left')
            final_df.to_csv(job_user_date_full_path, index=False)
            print(f"Finished creating {job_user_date_full_path}")
        except Exception as e:
            print(f"Error creating {job_user_date_full_path}: {e}")
    else:
        print(f"Skipping {job_user_date_full_path}: one or both of {job_user_date_path} or {slurm_log_path} not found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate local metadata files for MIT Supercloud dataset.")
    parser.add_argument("local_dataset_path", type=str, 
                        help="The root path to your locally downloaded MIT Supercloud dataset.")
    args = parser.parse_args()
    
    generate_local_metadata(args.local_dataset_path)
