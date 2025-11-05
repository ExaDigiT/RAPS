#!/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib
import numpy

import sys

matplotlib.rcParams['text.usetex'] = True

plt.style.use("paper.mplstyle")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Libertine"
})

plt.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Linux Libertine O"], # Specify the font family
})



pt = 1. / 72.27
width = 1.2*241.14749 * pt
golden = (1 + 5**0.5) / 2
height = width / golden * 4./5.
# COLUMNWIDTH241.14749pt TEXTWIDTH506.295pt


carray = []
t = plt.get_cmap('tab10').colors
for i in range(0,len(t)):
    carray.append(t[i])
g = carray[2]
carray[2] = carray[4]
carray[4] = g

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir>")
    exit()
# e.g. path = "$HOME/Repositories/exadigit/raps/simulation_results/frontier/nnew_fkg_2024-01-18"

policies = [
    'replay',
#    'replay-ffbf',
#    'fcfs-ffbf',
#    'priority-ffbf', # on fcfs
#    'sjf-ffbf',
#    'ljf-ffbf', # on prio
    'acct_avg_power-ffbf',
    'acct_low_avg_power-ffbf',
#    'acct_avg_power_w4lj-ffbf',
    'acct_edp-ffbf',
    'acct_fugaku_pts-ffbf',
    #'acct_ed2p-ffbf', #Sim to edp
    #'acct_pdp-ffbf',

]
#policies = ['fcfs-nobf','fcfs-easy','priority-nobf','priority-easy','priority-ffbf','replay']
#policies = ['fcfs-nobf','fcfs-easy','priority-ffbf','replay']
#policies = ['replay','prio-ffbf','fugaku_pts']
#files = ['cooling_model.parquet', 'loss_history.parquet', 'power_history.parquet', 'util.parquet']
#files = ['cooling_model.parquet', 'power_history.parquet', 'util.parquet']
#files = ['util.parquet', 'power_history.parquet', 'cooling_model.parquet']
files = ['util.parquet', 'power_history.parquet']
files = ['power_history.parquet']
#files = ['loss_history.parquet', 'power_history.parquet', 'util.parquet']
#files = ['power_history.parquet', 'util.parquet', 'cooling_model.parquet']

prefix = ""

policy_path = {f"{policy}":f"{path}/{prefix}{policy}" for policy in policies}
full_files = {f"{policy}":f"{path}/{prefix}{policy}/{file}" for policy in policies for file in files}


def iter_to_seconds(i):
    return i * 15


fig, axs = plt.subplots(len(files),figsize=(width, height*len(files)),sharex=True)
if isinstance(axs, matplotlib.axes._axes.Axes):
    axs = [axs]
elif isinstance(axs, numpy.ndarray):
    pass
else:
    pass

for i,file in enumerate(files):
    policy_files = [f"{path}/{prefix}{policy}/{file}" for policy in policies]
    for c,policy_file in enumerate(policy_files):
        # df = pd.read_parquet(policy_file)
        x = 'time'
        xlab = 'Time [hours]'
        policy = policy_file.split('/')[-2]
        if file == "power_history.parquet":
            y = 'power [kw]'
            ylab = 'Power [kW]'
            df = pd.read_parquet(policy_file)
            df = df.rename(columns={0:'time',1:'power [kw]'})


        elif file == "cooling_model.parquet":
            y = 'pue'
            ylab = 'PUE'

            df = pd.read_parquet(policy_file)
            df['index'] = df.index
            df[x] = df['index'].apply(iter_to_seconds)
            ymax = max(df['pue'])


        elif file == "loss_history.parquet":
            y = 'loss [kw]'
            ylab = 'Loss [kW]'

            df = pd.read_parquet(policy_file)
            df = df.rename(columns={0:'time',1:'loss [kw]'})

        elif file == "util.parquet":
            y = 'utilization'
            ylab = 'Utilization'

            df = pd.read_parquet(policy_file)
            df = df.rename(columns={0:'time', 1:'utilization [%]'})
            df[y] = df['utilization [%]'] / 100



        else:
            raise KeyError

        timeline_s = [0,21600,43200,64800,86400]
        timeline_h = ['0:00','6:00','12:00','18:00','24:00']
        axs[i].set_xticks(timeline_s,timeline_h)
        axs[i].set_xlabel(xlab)

        axs[i].plot(df[x],df[y], label=policy,
                    #linewidth=0.5,
                    marker='', color=carray[c])
        axs[i].set_ylabel(ylab)
        #$axs[i].plot(df[0],df[1],label=policy)
    if file == "power_history.parquet":
        axs[i].legend(loc='center left',bbox_to_anchor=(0.02, 0.6))
        axs[i].set_title('Power',x=0.1,y=0.80)
    elif file == "util.parquet":
        axs[i].set_title('Utilization')
        axs[i].legend(loc='lower left')
    elif file == "cooling_model.parquet":
        axs[i].set_title('PUE')
        axs[i].legend(loc='upper left')
    elif file == "loss_history.parquet":
        axs[i].set_title('Loss')
        axs[i].legend(loc='upper left')
    else:
        raise KeyError()
#plt.show()
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.subplots_adjust(hspace=0)
plt.tight_layout(pad=0,w_pad=0.0,h_pad=-0.08)#3)
plt.savefig(f"nnew_fkg_2024-01-18.png",bbox_inches='tight',pad_inches = 0.02, dpi = 300)#, bbox_inches='tight')
