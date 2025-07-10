#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:14:23 2024

@author: daf
"""

# Given a start and end date identify those jobs that occur in this range and then download them 
# from S3 into data/trace as a pcikle file (all traces will be in the same file)

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import pandas as pd
import numpy as np 
from io import StringIO
import pickle
from datetime import datetime
import shutil
import gzip
from scipy.sparse import csr_matrix as csr
import matplotlib.pyplot as plt

def main(): 
    
    # Get the directory of the current file
    mit_dir = os.path.dirname(os.path.abspath(__file__))
    src_data_dir = mit_dir + '/source_data'
    ################ Select correct files. 
    ################  CHANGE THESE 2 LINES ######################### 
    start_date = '21052021' # EU format day/month/year
    end_date = '22052021' 
    
    
                   
    ################# Load index files to use to look up the correct files. 
    # MIT S3 bucket address. 
    bucket_name = 'mit-supercloud-dataset'
    prefix = 'datacenter-challenge/202201/' 
    
    # Load the s3 file index file 
    fyle = mit_dir + '/source_data/file_list.csv'
    file_df = pd.read_csv(fyle,sep='\t')
    gpu_file_df = file_df[file_df['File Name'].str.contains('/gpu/')].copy()
    gpu_file_df['jobid'] = gpu_file_df['File Name'].str.extract(r'/([^/]+?)-')
    gpu_file_df['jobid'] = gpu_file_df['jobid'].astype(int)
    
    # Load the index file 
    fyle = mit_dir + '/source_data/job_user_date_full.csv'
    job_index_df = pd.read_csv(fyle)
    
    date_obj = datetime.fromtimestamp(job_index_df.start.min())
    date_min_str = date_obj.strftime('%d-%m-%Y')
    date_obj = datetime.fromtimestamp(job_index_df.start.max())
    date_max_str = date_obj.strftime('%d-%m-%Y')
    print('Data set contains data between: ' +date_min_str + ' and ' + date_max_str )
    
    # Create and clear the trace directory. 
    tracedir = mit_dir + '/data/trace/'
    if os.path.exists(tracedir):
        pass # do nothing - might want to change this later
        # shutil.rmtree(tracedir)  # Remove everything in the folder
        # os.makedirs(tracedir)    # Recreate the folder after clearing it
    else:
        os.makedirs(tracedir)
        
    
    st_date = datetime.strptime(start_date, '%d%m%Y')
    st_date = int(st_date.timestamp())
    en_date = datetime.strptime(end_date, '%d%m%Y')
    en_date = int(en_date.timestamp())
    
    if st_date < job_index_df.start.min(): 
        print('Warning: start date (' + start_date + ') is before the start of the dataset (' + date_min_str + ') ')
    if st_date > job_index_df.start.max(): 
        print('Error: start date (' + start_date + ') is after the end of the dataset (' + date_max_str + ') ')
        crashhere
    
    # find the jobs that start between start and end dates. 
    sift = (job_index_df.start > st_date) & (job_index_df.start < en_date)
    print('You have selected ' + str(sift.sum()) + ' fiies to download ')
    
    ##################### Download from S3 
    
    df = job_index_df[sift].copy()
    df['target']  =  df.filename.str.replace('-summary', '-timeseries')
    # Go through each file and download it 
    # Set up S3 client. 
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    cnt=0 
    
    for s in df.target:
        jobid = s.split('/')[-1].split('-')[0]
        jobid=int(jobid)
        fyle = os.path.join(mit_dir +'/data/trace/', s.split('/')[-1])
        
        # download the data unless we already have it
        if not os.path.exists(fyle):
            s3.download_file(bucket_name,  s, fyle)
        # See if there are GPU files for this job. 
        sift_gpu = gpu_file_df.jobid == jobid
        if sift_gpu.sum()>0: 
            gpu_fyles = gpu_file_df['File Name'][sift_gpu]
            for ss in gpu_fyles: 
                gfyle = os.path.join(mit_dir +'/data/trace/', ss.split('/')[-1])
                if not os.path.exists(gfyle):
                    s3.download_file(bucket_name,  ss, gfyle)
                   
        else:
            pass 
        cnt=cnt+1
        if cnt%50 ==0: 
            print('Downloaded ' + str(cnt) + ' of ' + str(sift.sum()))
    
    ##################### Process. 
    
    # Load the slurm log to grab additional attributes. 
    slurm_df = pd.read_csv(src_data_dir+'/slurm-log.csv')
    
    dfiles_raw = os.listdir(tracedir)
    # Sort so we process the cpu files first (we need the result for the gpu files)
    dfiles = sorted(dfiles_raw, key=lambda x: 'timeseries' not in x)
    dfiles = [file for file in dfiles if 'lock' not in file]
    
    print('Downloaded ' + str(len(dfiles)) + ' files. Processing ... ')
    L = len(dfiles)
    cnt = 0 
    data_dict = {}
    for s in dfiles:
        if cnt%100==0:
            print('processing file ' + str(cnt) + ' of ' + str(L))
        cnt = cnt+1
        fyle = os.path.join(mit_dir +'/data/trace/', s.split('/')[-1])
        dfi = pd.read_csv(fyle) 
        
        jobid = int(s.split('-')[0])
        if jobid not in data_dict.keys():
            data_dict[jobid] = {}
            # Add slurm data on creation 
            idx = np.where(slurm_df['id_job']==jobid)[0]
            if idx.shape[0]!=1: 
                crashhere
            else: 
                data_dict[jobid]  = slurm_df.iloc[idx[0]].to_dict()        
        if ('timeseries' in s) and ('lock' not in s): 
            if 'cpu' in data_dict[jobid].keys():
                print('error a job cant have more than one cpu traces')
                crashhere
            else: 
                cpu_ser = proc_cpu_series(dfi)
                data_dict[jobid]['cpu'] = cpu_ser
                
        elif 'gpu_index' in dfi.keys():
            mm = dfi.utilization_gpu_pct.max()
            print('GPU max: ' + str(mm) )
            # Get the gpu node and rack 
            rack = s.split('-')[1]
            node = s.split('-')[2].split('.csv')[0]
            cpu_df = data_dict[jobid]['cpu'] 
            
            
            if 'gpu' not in data_dict[jobid].keys():
                data_dict[jobid]['gpu'] = {}
                data_dict[jobid]['gpu_cnt']=0 
                data_dict[jobid]['grack']=[rack]   
                data_dict[jobid]['gnode']=[node]
                gpu_cnt = data_dict[jobid]['gpu_cnt']
                gpu_ser,gpu_cnt = proc_gpu_series(cpu_df,dfi,gpu_cnt)
                data_dict[jobid]['gpu'] = gpu_ser
            else: 
                data_dict[jobid]['grack'].append(rack)   
                data_dict[jobid]['gnode'].append(node)
                gpu_df = data_dict[jobid]['gpu']
                gpu_cnt =  data_dict[jobid]['gpu_cnt']
                gpu_ser,gpu_cnt = proc_gpu_series(cpu_df,dfi,gpu_cnt)
                # combine with the existing df 
                df_merged = pd.merge(gpu_df, gpu_ser, on='utime') 
                asds
                if df_merged.shape[0] != gpu_df.shape[0]:
                    crashhere
                data_dict[jobid]['gpu'] = df_merged
            data_dict[jobid]['gpu_cnt']= gpu_cnt
               
    asd
    # save to pickle file.
    pkldir =  mit_dir + '/data/pkl/'
    os.makedirs(pkldir, exist_ok=True)
    t1 = datetime.fromtimestamp(st_date)
    tf1 = t1.strftime('%d_%m_%Y')
    t2 = datetime.fromtimestamp(en_date)
    tf2 = t2.strftime('%d_%m_%Y')
    fyle = 'data_' + tf1 + '__'+tf2 
    fyle = pkldir+fyle +'.pkl.gz'
    with gzip.open(fyle, 'wb') as file:
        pickle.dump(data_dict, file)
    
    return 

def proc_gpu_series(cpu_df,dfi,gpu_cnt):
    # Process GPU series by interpolating it to the same times as the cpu series. 
    
    # time checks 
    t_cpu = np.array([cpu_df.utime.min(), cpu_df.utime.max() , 0])
    t_cpu[2]=t_cpu[1]-t_cpu[0]
    t_gpu = np.array([dfi.timestamp.astype(int).min(), dfi.timestamp.astype(int).max(),0])
    t_gpu[2]=t_gpu[1]-t_gpu[0]
    
    dcpu = pd.to_datetime(t_cpu, unit='s')
    dgpu = pd.to_datetime(t_gpu, unit='s')
    t1 = (dcpu[1]-dcpu[0]).total_seconds()
    t2 = (dgpu[1]-dgpu[0]).total_seconds()
    per_dif = (t1-t2)/t2*100
    print(per_dif)
    if abs(per_dif) > 10: 
        # More than 2% difference in the time taken, halt and look at it
        crashhere
    
    # So move the GPU time to the CPU times. 
    dfi['t_fixed'] = dfi.timestamp-dfi.timestamp.min()+t_cpu[0]

    ugpus = dfi.gpu_index.unique()
    gpu_df= pd.DataFrame({'utime':  cpu_df['utime'].values})
    
    
    for u in ugpus: 
        dfg = dfi[dfi.gpu_index==u].copy()
        
        # Perform an interpolation
        fylds = ['gpu_index', 'utilization_gpu_pct',
               'utilization_memory_pct', 'memory_free_MiB', 'memory_used_MiB',
               'temperature_gpu', 'temperature_memory', 'power_draw_W']
        
        
        
        for ff in fylds: 
            x1 = dfg['t_fixed'].values
            y1 = dfg[ff].values
            xv = cpu_df['utime'].values
        
            # Interpolate using NumPy
            yv = np.interp(xv, x1, y1)
        
            gpu_df[ff] = yv
            ss  =  str(gpu_cnt)
            ren = {'utilization_gpu_pct': 'gpu_' + ss, 
                   'utilization_memory_pct': 'gpu_mem_' + ss,
                   'temperature_gpu': 'gpu_temp_' + ss, 
                   'power_draw_W':'gpu_p_'+ ss,
                   }            
        gpu_df.rename(columns=ren, inplace=True)
        gpu_cnt = gpu_cnt + 1
    
    return gpu_df,gpu_cnt

def proc_cpu_series(dfi): 
    # This is the code that processes cpu data and performs the following steps: 
    # 1. Remove information from step [-1,-4] as these are empty. 
    # 2. give outliers their nearest neighbour values. There are spikes of outliers in the utilsation, I think thw whole row is rotten too. They are values like 40000
    # 3. For each series get the max cpu utilisation at each time step. 
    #   Save these for the output. 
    # 4. Get the average cpu utilsation per series (maxed from step 3) 
    
    # 1  Remove information from step [-1,-4] as these are empty. 
    sift = dfi.Step.isin([-1,-4,'-1','-4'])
    if dfi.CPUUtilization[sift].sum() >0: 
        print('found a series that breaks the rule, check it')
        # The -1 -4 indicators should be for non-events. IF the cpu utilisation has values something is up, might be a spike or something but the rule needs to be changed. 
        crashhere
    # remove
    dfi = dfi[~sift].copy()
    
    # Check for 1-1 series node correspondences and if not then there is an issue we need to clean up. 
    if False: 
        unode_series = dfi.groupby(['Node', 'Series']).size().reset_index(name='count')
        unode = dfi.Node.unique()
        for n in unode: 
            sift = dfi.Node == n 
            splits = dfi[sift].groupby('Series').size().reset_index(name='count')
            splits = splits.sort_values(by='count', ascending=False)
            for i in range(splits.shape[0]): 
                # Reassign the Series number back to the max for the node. 
                if i==0: 
                    dest_ser = splits.iloc[i].Series
                else: 
                    # reassign the targets. 
                    faulty_ser = splits.iloc[i].Series
                    sift_reas  = sift & (dfi.Series ==faulty_ser )
                    dfi.loc[sift,'Series'] = dest_ser
                    if sift_reas.sum()>40: 
                        asd
                    print('Reassigning ' + str(sift_reas.sum()) + ' rows with faulty series values (from a total of  ' + str(splits['count'][0])+ ' )')
    t = pd.to_datetime(dfi.EpochTime, unit='s')
    start_time = t.min()
    steps = (t - start_time).dt.total_seconds() // 10
    # Convert to integer type if needed
    steps = steps.astype(int)
    dfi['t']= steps

    sid, uniques = pd.factorize(dfi.Step)
    dfi['sid']= sid
    

    
    # 2. Outliers. 
    sift = (dfi.CPUUtilization > 500) & (dfi.CPUUtilization < 600)
        # Clip these back to 500
    if sift.sum()>0: 
        asd
        print('clipping ' + str(sift.sum()) + ' values' )
        dfi.loc[sift, 'CPUUtilization'] = 500
        
    # select rows with >600 as outliers.
    sift = dfi.CPUUtilization > 600
    if sum(sift)>0: 
        # Set to the nearest value less than 600. 
        dfi.loc[sift, 'CPUUtilization'] = dfi['CPUUtilization'].where(~sift).ffill().combine_first(dfi['CPUUtilization']).where(dfi['CPUUtilization'] <= 600)   
    
    # 3. There are multiple series so we want to get the maximum (as only one series at a time is active)
    useries = dfi.Series.unique()
    inds = np.arange(dfi.t.max()+1)
    # Create a data frame to hold the results.
    df = pd.DataFrame({'t':inds})
    Xm = np.zeros((len(useries),inds.shape[0]))
    Xrss = np.zeros((len(useries),inds.shape[0]))
    Xvm = np.zeros((len(useries),inds.shape[0]))
    Xreadmb = np.zeros((len(useries),inds.shape[0]))
    Xwritemb = np.zeros((len(useries),inds.shape[0]))

    cnt=0
    for i in useries: 
        sift = dfi.Series == i 
        M = len(inds)
        N = dfi.sid[sift].max()+1
        # create a #series x #time steps csr then max it to get the actual readings. 
        X = csr( (dfi.CPUUtilization[sift],(dfi.t[sift],dfi.sid[sift])),shape = (M,N) )
        mm = np.array(X.max(axis=1).todense()).reshape(-1,)
        df['cpu_' + str(i)] = mm 
        Xm[cnt,:] = mm 
        
        # RSS 
        X = csr( (dfi.RSS[sift],(dfi.t[sift],dfi.sid[sift])), shape = (M,N) )
        mm = np.array(X.max(axis=1).todense()).reshape(-1,)
        df['rss_' + str(i)] = mm 
        Xrss[cnt,:] = mm 
        
        # VMsize 
        X = csr( (dfi.VMSize[sift],(dfi.t[sift],dfi.sid[sift])), shape = (M,N) )
        mm = np.array(X.max(axis=1).todense()).reshape(-1,)
        df['vm_' + str(i)] = mm 
        Xvm[cnt,:] = mm 

        # ReadMB 
        X = csr( (dfi.ReadMB[sift],(dfi.t[sift],dfi.sid[sift])), shape = (M,N) )
        mm = np.array(X.max(axis=1).todense()).reshape(-1,)
        df['readmb_' + str(i)] = mm 
        Xreadmb[cnt,:] = mm 

        # WriteMB 
        X = csr( (dfi.WriteMB[sift],(dfi.t[sift],dfi.sid[sift])), shape = (M,N) )
        mm = np.array(X.max(axis=1).todense()).reshape(-1,)
        df['writemb_' + str(i)] = mm 
        Xwritemb[cnt,:] = mm 

        
        cnt=cnt+1
        
    df['cpu_utilisation'] = Xm.mean(axis=0)
    df['rss'] = Xrss.sum(axis=0)
    df['vm'] = Xvm.sum(axis=0)
    df['readmb'] = Xreadmb.sum(axis=0)
    df['writemb'] = Xwritemb.sum(axis=0)
    
    
    df['timestamp'] = start_time + pd.to_timedelta(df.t * 10, unit='s')
    df['utime'] = df['timestamp'].astype('int64') // 10**9
    return df 

if __name__ == "__main__":
    main()
    