#!/usr/bin/env python3
"""
Single‑script tool to:
  1) Ensure slurm-log.csv is present locally (download if missing)
  2) Filter jobs by submit date and classify CPU vs GPU by gres/tres
  3) Build or load a one‑time S3 manifest of trace keys (CPU & GPU)
  4) Filter that manifest by job IDs and download matching files

Usage:
  python download_data.py [--start DDMMYYYY] [--end DDMMYYYY] \
                        [--partition all|part-cpu|part-gpu] \
                        [--outdir PATH] [--max-jobs N] [--dry-run]

Defaults:
  --start 21052021   # 21 May 2021 (inclusive)
  --end   22052021   # 22 May 2021 (exclusive)

Flags:
  --max-jobs N       # Only process first N jobs (for testing)
  --dry-run          # List a sample of files without downloading
"""
# Suppress urllib3 InsecureRequestWarning
from .utils import (
    load_slurm_log,
    build_or_load_manifest,
    # filter_keys_by_jobs  # Defined below! not in utils...
)
from tqdm import tqdm
from botocore.client import Config
from botocore import UNSIGNED
import boto3
import pandas as pd
from datetime import datetime
import re
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Default date window
DEFAULT_START = "21052021"
DEFAULT_END = "22052021"


def ensure_slurm_log(s3, bucket, key, dest):
    if os.path.exists(dest):
        print(f"Found existing slurm-log.csv at {dest}, skipping download.")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading slurm-log.csv → {dest}")
    s3.download_file(bucket, key, dest)
    print("Downloaded slurm-log.csv.")


def list_and_filter_jobs(csv_path, start, end, partition):
    df = pd.read_csv(csv_path)
    df['time_submit'] = pd.to_datetime(df['time_submit'], unit='s')

    dt0 = datetime.strptime(start, "%d%m%Y")
    dt1 = datetime.strptime(end,   "%d%m%Y")
    window = df[(df['time_submit'] >= dt0) & (df['time_submit'] < dt1)]

    # Identify GPU-using jobs via gres_used or tres_alloc
    gres = window['gres_used'].fillna("").astype(str)
    tres = window['tres_alloc'].fillna("").astype(str)
    gpu_jobs = set(window.loc[
        gres.str.contains("gpu", case=False) |
        tres.str.contains(r"(?:1001|1002)=", regex=True),
        'id_job'
    ])

    if partition == 'part-gpu':
        job_ids = sorted(gpu_jobs)
    elif partition == 'part-cpu':
        job_ids = sorted(set(window['id_job']) - gpu_jobs)
    else:
        job_ids = sorted(window['id_job'].unique())

    print(f"Selected {len(job_ids)} jobs from {dt0.date()} to {dt1.date()} on {partition}.")
    return job_ids


def build_manifest(s3, bucket, prefix, manifest_path):
    """
    One-time listing of all CSV keys under cpu/ and gpu/ prefixes.
    Writes each key to manifest_path.
    """
    cpu_pref = prefix + 'cpu/'
    gpu_pref = prefix + 'gpu/'
    paginator = s3.get_paginator('list_objects_v2')
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as mf:
        # CPU
        print("Building manifest: listing CPU keys...")
        for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=cpu_pref), desc="CPU pages", unit="page"):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    mf.write(key + '\n')
        # GPU
        print("Building manifest: listing GPU keys...")
        for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=gpu_pref), desc="GPU pages", unit="page"):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    mf.write(key + '\n')
    print(f"Manifest written to {manifest_path}.")


# def load_manifest(manifest_path):
#    with open(manifest_path) as f:
#        return [line.strip() for line in f]


def filter_keys_by_jobs(keys, job_ids):
    """
    Parse job ID from start of filename (<jobid>-...) or via -r<jobid>- in GPU names,
    keep only keys matching job_ids.
    """
    sel = []
    for key in keys:
        fname = os.path.basename(key)
        # Try CPU style: '<jobid>-'
        parts = fname.split('-', 1)
        jid = None
        try:
            jid = int(parts[0])
        except ValueError:
            # Try GPU style: '-r<jobid>-'
            m = re.search(r'-r(\d+)-', fname)
            if m:
                jid = int(m.group(1))
        if jid and jid in job_ids:
            sel.append(key)
    return sel


def download_traces(s3, bucket, prefix, outdir, keys, dry_run):
    if dry_run:
        print("Dry-run: sample of matching keys:")
        for key in keys[:10]:
            print("  ", key)
        return
    for key in tqdm(keys, desc="Downloading traces"):
        rel = key[len(prefix):]
        dest = os.path.join(outdir, rel)
        if os.path.exists(dest):
            tqdm.write(f"Warning: {dest} exists, skipping.")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        s3.download_file(bucket, key, dest)
    print("All requested traces downloaded.")


def download(args):
    """
    Subcommand entrypoint for 'mit_supercloud download'.
    Downloads slurm-log.csv and all matching CPU/GPU trace files from S3.
    """
    # 1) Initialize anonymous S3 client with SSL verification disabled
    s3 = boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED),
        verify=False
    )

    # 2) Ensure local copy of slurm-log.csv
    slurm_key = f"{args.prefix}slurm-log.csv"
    slurm_path = os.path.join(args.outdir, 'slurm-log.csv')
    ensure_slurm_log(s3, args.bucket, slurm_key, slurm_path)

    # 3) Load and filter SLURM log to determine CPU/GPU job sets
    _, cpu_jobs, gpu_jobs = load_slurm_log(
        slurm_path,
        args.start,
        args.end
    )
    if args.partition == 'part-cpu':
        job_ids = cpu_jobs
    elif args.partition == 'part-gpu':
        job_ids = gpu_jobs
    else:
        job_ids = cpu_jobs | gpu_jobs
    if args.max_jobs:
        job_ids = set(list(job_ids)[:args.max_jobs])
    print(f"Processing {len(job_ids)} jobs (partition={args.partition})")

    # 4) Build or load the one-time manifest of all trace keys
    manifest_path = os.path.join(args.outdir, 'file_manifest.txt')
    all_keys = build_or_load_manifest(
        s3, args.bucket, args.prefix, manifest_path
    )

    # 5) Filter manifest to only the trace keys for our job IDs
    trace_keys = filter_keys_by_jobs(all_keys, job_ids)
    cpu_count = sum(1 for k in trace_keys if k.startswith(f"{args.prefix}cpu/"))
    gpu_count = len(trace_keys) - cpu_count
    print(f"Total matching trace files: {len(trace_keys)} (CPU: {cpu_count}, GPU: {gpu_count})")

    # 6) Download or dry-run
    download_traces(
        s3,
        args.bucket,
        args.prefix,
        args.outdir,
        trace_keys,
        args.dry_run
    )
