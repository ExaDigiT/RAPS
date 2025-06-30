#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:14:23 2024

@author: daf
"""


import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

# Get the directory of the current file
mit_dir = os.path.dirname(os.path.abspath(__file__))



def list_s3_files_and_sizes(bucket_name, prefix=''):
    # Initialize an S3 client with no signing
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    file_names = []
    file_sizes_gb = []

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                namm = obj['Key']
                file_names.append(namm)
                file_size_gb = obj['Size'] / (1024 ** 2)  # Convert from bytes to GB
                file_sizes_gb.append(file_size_gb)
                print(f"{namm}: {file_size_gb:.4f} MB")
                
    return file_names, file_sizes_gb

def download_s3_bucket(bucket_name, prefix, datadir):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')

    # Recursively download all files with the given prefix
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                # Get the file's S3 key
                s3_key = obj['Key']
                local_file_path = os.path.join(datadir, s3_key)
                local_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                print(f"Downloading {s3_key} to {local_file_path}...")
                s3.download_file(bucket_name, s3_key, local_file_path)

# Replace 'your-bucket-name' with the actual S3 bucket name
bucket_name = 'mit-supercloud-dataset'
prefix = 'datacenter-challenge/202201/' 

# Get the list of file names and sizes
file_names, file_sizes_gb = list_s3_files_and_sizes(bucket_name, prefix)


# download one cpu and 1 gpu of data. 
bucket_name = 'mit-supercloud-dataset'
subfolder = 'datacenter-challenge/202201/cpu/0026/'
datadir = mit_dir + '/source_data'
download_s3_bucket(bucket_name, subfolder, datadir)

subfolder = 'datacenter-challenge/202201/gpu/0020/'
download_s3_bucket(bucket_name, subfolder, datadir)


# Output the results
print("Files in S3 bucket:")
for name, size in zip(file_names, file_sizes_gb):
    print(f"{name}: {size:.2f} GB")

# Example: You can use the lists for further processing
# file_names -> list of file paths
# file_sizes_gb -> list of file sizes in GB