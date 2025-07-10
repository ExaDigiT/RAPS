#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:14:23 2024

@author: daf
"""
# This script will look to see if you have certain files and if not it will create/download them (this avoids large unneccesary downloads)
# In addition it is used to download data for certain date ranges that you can specify (across all machines). 
# To set the date ranges change start_date and end_date on lines 

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import pandas as pd
from io import StringIO

# Get the directory of the current file
mit_dir = os.path.dirname(os.path.abspath(__file__))

start_date = '01012020' # EU format day/month/year
end_date = '01012020' 
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
                s3_stem = s3_key[28:]
                local_file_path = os.path.join(datadir, s3_stem)
                
                local_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                print(f"Downloading {s3_key} to {local_file_path}...")
                s3.download_file(bucket_name, s3_key, local_file_path)

def index_summary_file(bucket_name, prefix, datadir):
    paginator = s3.get_paginator('list_objects_v2')
    results = []
    # Check if the bucket contains any objects
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Check if the key ends with '-summary.csv'
                if key.endswith('-summary.csv'):
                    # Read the CSV file from S3 into a DataFrame
                    csv_obj = s3.get_object(Bucket=bucket_name, Key=key)
                    body = csv_obj['Body'].read().decode('utf-8')  # Decode bytes to string
                    
                    # Use StringIO to read the CSV data
                    df = pd.read_csv(StringIO(body))
                    
                    # Get the maximum value from the 'epoch' column
                    st_time = df['Min_EpochTime'].min()
                    ed_time = df['Max_EpochTime'].max()
                    node_count = df.Node.unique().shape[0]
                    jobid = int(key.split('/')[-1].split('-')[0])
                    # Append the results to the DataFrame
                    results.append({'job_id': jobid, 'filename': key, 'start': st_time, 'end' : ed_time, 'node_count': node_count})
                    print(f"Processed: {key}")
                    
    df = pd.DataFrame(results)
    return df
               
asd
# MIT S3 bucket address. 
bucket_name = 'mit-supercloud-dataset'
prefix = 'datacenter-challenge/202201/' 


# Get the list of S3 file names and sizes and save (unless its already there)
fyle = mit_dir + '/source_data/file_list.csv'

# Check if file exists
if not os.path.exists(fyle):
    file_names, file_sizes_gb = list_s3_files_and_sizes(bucket_name, prefix)
    # Open a file in write mode
    with open(fyle, "w") as file:
        # Write the header (optional)
        file.write("File Name\tSize (MB)\n")    
        # Iterate over both lists and write each file name and its size
        for name, size in zip(file_names, file_sizes_gb):
            file.write(f"{name}\t{size:.2f} \n")

# Download the following root dir files. 
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
dfiles = ['LICENSE','README.md','labelled_job_stats.csv','labelled_jobids.csv'
          ,'node-data.csv','slurm-log.csv','tres-mapping.txt']

for s in dfiles:
    fyle = os.path.join(mit_dir +'/source_data', os.path.basename(s))
    if not os.path.exists(fyle):
        s3.download_file(bucket_name, prefix + s, fyle)

# download one cpu and 1 gpu of data. 
bucket_name = 'mit-supercloud-dataset'
subfolder = 'datacenter-challenge/202201/cpu/0026/'
datadir = mit_dir + '/source_data'
#download_s3_bucket(bucket_name, subfolder, datadir)

subfolder = 'datacenter-challenge/202201/gpu/0020/'
#download_s3_bucket(bucket_name, subfolder, datadir)


# Create the job-user-date index file if it doesnt exist already. 
fyle = mit_dir + '/source_data/job_user_date.csv'
# Check if file exists
if not os.path.exists(fyle):
    print('This can take about 24 hours to complete.')
    job_index_df = index_summary_file(bucket_name, prefix, datadir)
    job_index_df.to_csv(fyle, index=False)
else: 
    job_index_df = pd.read_csv(fyle)
    
fyle = mit_dir + '/source_data/job_user_date_full.csv'
if not os.path.exists(fyle):
    # Open the slurm log to get the user id for each job. 
    slurm_df = pd.read_csv(mit_dir + '/source_data/slurm-log.csv')
    # Cut out all but the user job mapping 
    slurm_df = slurm_df[['id_job','id_user']]

    final_df = job_index_df.merge(slurm_df, left_on='job_id', right_on='id_job', how='left')
    final_df.to_csv(fyle, index=False)

print('Pre-processing to create an index linking jobs and users to dates is now complete and can be found in the file ')
print(fyle)




