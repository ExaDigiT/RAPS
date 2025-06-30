#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:18:26 2024

@author: daf
"""

# Download the paper describing the data 
import requests
import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import pandas as pd
from io import StringIO


############### Dir setup
# Get the directory of the current file
mit_dir = os.path.dirname(os.path.abspath(__file__))

# Create a local directory structure 
dirs = ['source_data','papers']
for s in dirs:
    local_dir = mit_dir + '/'+s
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

# URL of the PDF file
url = 'https://arxiv.org/pdf/2108.02037'

# Send a GET request to the URL
response = requests.get(url)
# Check if the request was successful
if response.status_code == 200:
    # Specify the local filename to save
    pdf_filename = mit_dir + '/papers/2108.02037.pdf'
    

    # Write the content to a local file
    with open(pdf_filename, 'wb') as file:
        file.write(response.content)
        
# Download the summary data only from the server to get the dates for each trace. 

############### Create an index file to allow us to select jobs by date. 


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
               
# MIT S3 bucket address. 
bucket_name = 'mit-supercloud-dataset'
prefix = 'datacenter-challenge/202201/' 

# Get the list of S3 file names and sizes and save (unless its already there)
fyle = mit_dir + '/source_data/file_list.csv'

# Create the file list if its not there already
if not os.path.exists(fyle):
    file_names, file_sizes_gb = list_s3_files_and_sizes(bucket_name, prefix)
    # Open a file in write mode
    with open(fyle, "w") as file:
        # Write the header (optional)
        file.write("File Name\tSize (MB)\n")    
        # Iterate over both lists and write each file name and its size
        for name, size in zip(file_names, file_sizes_gb):
            file.write(f"{name}\t{size:.2f} \n")

# Download the following root dir files (always)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
dfiles = ['LICENSE','README.md','labelled_job_stats.csv','labelled_jobids.csv'
          ,'node-data.csv','slurm-log.csv','tres-mapping.txt']

for s in dfiles:
    fyle = os.path.join(mit_dir +'/source_data', os.path.basename(s))
    if not os.path.exists(fyle):
        s3.download_file(bucket_name, prefix + s, fyle)
        
# Create the job-user-date index file if it doesnt exist already. 
datadir = mit_dir + '/source_data'
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


print('The MIT supercloud is now set up, the paper describing the dataset can be found in /papers')
print('The slurm-log and node data has been downloaded. However no cpu or gpu job traces have been downloaded. As there are 2TB of these we have created a script called create_trace.py to allow you to download and select a subset of the data dependent on time.')

