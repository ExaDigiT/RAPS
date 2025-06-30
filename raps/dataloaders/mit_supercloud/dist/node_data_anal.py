#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:46:42 2024

@author: daf
"""

# Analyse the node data. 

import pandas as pd
import numpy as np
import os,subprocess

# Get the directory of the current file
mit_dir = os.path.dirname(os.path.abspath(__file__))

node_fyle  = mit_dir+'/source_data/node-data.csv'

# Define a function to skip rows that are not multiples of 4


# Calculate the total number of rows in the file (optional, to improve efficiency)
Nr = sum(1 for row in open(node_fyle)) # 34M rows. 
K=100 # Reduction factor. 
keep_rows = np.arange(3, Nr, K)

temp_fyle = node_fyle[:-13] + 'temp.csv' 
cmd = f"awk 'NR == 1 || NR % {K} == 0' \"{node_fyle}\" > \"{temp_fyle}\""


# Run the awk command using subprocess
subprocess.run(cmd, shell=True, check=True)


# Read the CSV file, skipping rows that are not multiples of 4
df = pd.read_csv(temp_fyle)
df['datetime'] = pd.to_datetime(df['Time'], unit='s')

# Display the resulting DataFrame
print(df) 