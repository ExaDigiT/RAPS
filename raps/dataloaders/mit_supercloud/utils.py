import os
import re
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.client import Config


def load_slurm_log(slurm_path: str, start_date: str, end_date: str):
    """
    Load Slurm log and filter jobs by submission window.

    Args:
        slurm_path: Path to local slurm-log.csv
        start_date: "DDMMYYYY" inclusive start
        end_date:   "DDMMYYYY" exclusive end

    Returns:
        tuple(
            pandas.DataFrame filtered on date window,
            set of CPU-only job IDs,
            set of GPU-using job IDs
        )
    """
    df = pd.read_csv(slurm_path)
    # Convert submit times
    df['time_submit'] = pd.to_datetime(df['time_submit'], unit='s')
    dt0 = datetime.strptime(start_date, "%d%m%Y")
    dt1 = datetime.strptime(end_date,   "%d%m%Y")
    window = df[(df['time_submit'] >= dt0) & (df['time_submit'] < dt1)]

    # Detect GPU jobs via gres_used or tres_alloc
    gres = window['gres_used'].fillna("").astype(str)
    tres = window['tres_alloc'].fillna("").astype(str)
    gpu_jobs = set(
        window.loc[
            gres.str.contains("gpu", case=False) |
            tres.str.contains(r"(?:1001|1002)=", regex=True),
            'id_job'
        ]
    )
    cpu_jobs = set(window['id_job']) - gpu_jobs
    return window, cpu_jobs, gpu_jobs


def build_or_load_manifest(s3, bucket: str, prefix: str, manifest_path: str):
    """
    Build a one-time manifest of all .csv keys under cpu/ and gpu/ in S3,
    or load an existing manifest from disk.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 dataset root prefix (e.g. "datacenter-challenge/202201/")
        manifest_path: local path to cache the manifest

    Returns:
        List[str]: all S3 keys ending in .csv under cpu/ and gpu/
    """
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return [line.strip() for line in f]

    # Otherwise build manifest
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for kind in ('cpu', 'gpu'):
        pfx = prefix + f"{kind}/"
        for page in tqdm(
            paginator.paginate(Bucket=bucket, Prefix=pfx),
            desc=f"Listing {kind} pages", unit="page"
        ):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    keys.append(key)
    # Cache on disk
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        for key in keys:
            f.write(key + '\n')
    return keys


def filter_keys_by_jobs(all_keys: list, job_ids: set):
    """
    Filter a list of S3 keys to those belonging to specified job IDs.

    Args:
        all_keys: list of S3 keys from manifest
        job_ids:  set of job IDs (int)

    Returns:
        List[str] of keys matching CPU or GPU jobs
    """
    selected = []
    gpu_pattern = re.compile(r'-r(\d+)-')
    for key in all_keys:
        # CPU keys: prefix/jobid-...-timeseries.csv or -summary.csv
        if '/cpu/' in key:
            fname = os.path.basename(key)
            parts = fname.split('-', 1)
            try:
                jid = int(parts[0])
            except ValueError:
                continue
            if jid in job_ids:
                selected.append(key)
        # GPU keys: detect -r<jobid>- in filename
        elif '/gpu/' in key:
            fname = os.path.basename(key)
            m = gpu_pattern.search(fname)
            if m and int(m.group(1)) in job_ids:
                selected.append(key)
    return selected
