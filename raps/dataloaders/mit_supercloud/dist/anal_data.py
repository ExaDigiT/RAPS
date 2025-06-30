#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:49:04 2024

@author: daf
"""
import gzip 
import pickle 
import os 

# Get the directory of the current file
mit_dir = os.path.dirname(os.path.abspath(__file__))

# List the data files you want analysed into this list. It is assumed they live in /data/pkl
data_fyles = ['data_21_05_2021__22_05_2021.pkl.gz']

data = {}
# Combine the pickle files for comparison. 
for s in data_fyles: 
    fyle = mit_dir+'/data/pkl/' + s
    with gzip.open(fyle, 'rb') as file:
        datai = pickle.load(file)
    if data.keys is None: 
        data = datai
    else:
        # Check for common keys first 
        common_keys = list(data.keys() & datai.keys())
        if len(common_keys)>0: 
            print('Warning: there seems to be jobs overlapping in the data sets')
        
        # Combine
        data = {**data, **datai}

# Lets see how the job time series actually look

