#!/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

import sys



if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir>")
    exit()

# e.g. path = "$HOME/Repositories/exadigit/raps/simulation_results/b803010"

files = ['cooling_model.parquet', 'loss_history.parquet', 'power_history.parquet', 'util.parquet']

full_files = [f"{path}/{file}" for file in files]


def iter_to_seconds(i):
    return i * 15


SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



for i in [1]:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    power = path + "/" + files[2]
    loss = path + "/" + files[1]
    util = path + "/" + files[3]
    cooling = path + "/" + files[0]

    df_power = pd.read_parquet(power)
    df_power = df_power.rename(columns={0:'time',1:'power [kw]'})
    ax1.plot(df_power['time'],df_power['power [kw]'], color='black', label='Power kW]')

    df_loss = pd.read_parquet(loss)
    df_loss = df_loss.rename(columns={0:'time',1:'loss [kw]'})
    ax1.plot(df_loss['time'],df_loss['loss [kw]'], color='red', label='Loss [kW]')

    ax2 = ax1.twinx()

    df_cooling = pd.read_parquet(cooling)
    df_cooling['index'] = df_cooling.index
    df_cooling['time'] = df_cooling['index'].apply(iter_to_seconds)
    ymax = max(df_cooling['pue'])
    ax2.plot(df_cooling['time'],df_cooling['pue'], color='blue', label='PUE')

    df_util = pd.read_parquet(util)
    df_util = df_util.rename(columns={0:'time', 1:'utilization [%]'})
    df_util['utilization'] = df_util['utilization [%]'] / 100
    ax2.plot(df_util['time'],df_util['utilization'], color='orange', label='Utilization')

    ymax = max(max(df_cooling['pue']),max(df_util['utilization']))
    #ymax = max(0,max(df_util['utilization']))
    ax2.set_ylim([0, ymax * 1.05])

    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('[kW]')
    ax2.set_ylabel('[%]')
    #path
    #plt.title(path)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    #plt.rcParams.update({'font.size': 30})
    #plt.show()
    plt.savefig("test.png")
