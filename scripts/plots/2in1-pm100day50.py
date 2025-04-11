#!/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

import sys

import matplotlib
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
width = 1.2 * 241.14749 * pt
golden = (1 + 5**0.5) / 2
height = width / golden * 3. / 5.


carray = ['tab:cyan','tab:orange','tab:brown','tab:blue']

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print(f"Usage: python {sys.argv[0]} <simulation_result/dir>")
    exit()

# e.g. path = "$HOME/Repositories/exadigit/raps/simulation_results/marconi100/day51"

policies = ['fcfs-nobf','fcfs-easy','priority-nobf','priority-easy','priority-ffbf','replay']
policies = ['fcfs-nobf','fcfs-easy','priority-ffbf','replay']
files = ['cooling_model.parquet', 'loss_history.parquet', 'power_history.parquet', 'util.parquet']
files = ['power_history.parquet', 'util.parquet']
#files = ['util.parquet', 'power_history.parquet']
#files = ['loss_history.parquet', 'power_history.parquet', 'util.parquet']
#files = ['power_history.parquet', 'util.parquet', 'cooling_model.parquet']

policy_path = {f"{policy}":f"{path}/{policy}" for policy in policies}
full_files = {f"{policy}":f"{path}/{policy}/{file}" for policy in policies for file in files}


def iter_to_seconds(i):
    return i * 15

c_cnt=0
fig, axs = plt.subplots(len(files),figsize=(width,2 * height))
for i,file in enumerate(files):
    policy_files = [f"{path}/{policy}/{file}" for policy in policies]
    for c,policy_file in enumerate(policy_files):
        # df = pd.read_parquet(policy_file)
        x = 'time'
        xlab = 'Time [hours/days]'
        policy = policy_file.split('/')[-2]
        if file == "power_history.parquet":
            y = 'power [kw]'
            ylab = 'Power [kW]'
            #ymax = 26000
            #ymin = 6500
            #axs[i].set_ylim(ymin,ymax)
            df = pd.read_parquet(policy_file)
            df = df.rename(columns={0:'time',1:'power [kw]'})

        elif file == "cooling_model.parquet":
            if c_cnt == 0:
                y = 'pue'
                ylab = 'PUE'
            if c_cnt == 1:
                y = 'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.T_fac_ctw_r_C'
                ylab = 'Temperature [°C]'
            if c_cnt == 2:
                y = 'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.T_fac_ctw_s_C'
                ylab = 'Temperature [°C]'

            df = pd.read_parquet(policy_file)
            df['index'] = df.index
            df[x] = df['index'].apply(iter_to_seconds)
            ymax = max(df[y])
                #axs[i].plot(df[x],df[y], label=ylab)
                #y = 'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.T_fac_ctw_r_C'

        elif file == "loss_history.parquet":
            y = 'loss [kw]'
            ylab = 'Loss [kW]'
            #ylim = 29000
            #axs[i].set_ylim(0,ylim)

            df = pd.read_parquet(policy_file)
            df = df.rename(columns={0:'time',1:'loss [kw]'})
            #axs[i].plot(df[x],df[y], label=ylab)

        elif file == "util.parquet":
            y = 'utilization'
            ylab = r'Utilization [\%]'

            df = pd.read_parquet(policy_file)
            df = df.rename(columns={0:'time', 1:'utilization [%]'})
            df[y] = df['utilization [%]'] / 100
            #axs[i].plot(df[x], df[y], label=ylab)

        else:
            raise KeyError

        timeline_s = []
        timeline_text = []

        timeline_s.extend([4320000,4330800,4341600,4352400,4363200,4374000,4384800,4395600])
        timeline_text.extend(['0:00\nDay 50','3:00','6:00','9:00','12:00','15:00','18:00','21:00'])
        timeline_s.extend([4406400,4417200,4428000,4438800,4449600,4460400,4471200,4482000])
        timeline_text.extend(['0:00\nDay 51','3:00','6:00','9:00','12:00','15:00','18:00','21:00'])

        axs[i].set_xticks(timeline_s,timeline_text)
        if i == 1:
            pass
        else:
            axs[i].set_xticklabels([])  # Remove x-axis labels
            xlab = None

        axs[i].set_xlabel(xlab)
        axs[i].plot(df[x],df[y], label=policy, color=carray[c])
        axs[i].set_ylabel(ylab)
        #$axs[i].plot(df[0],df[1],label=policy)
    if file == "power_history.parquet":
        axs[i].legend(loc='lower right',frameon=True)
        axs[i].get_legend().get_frame().set_linewidth(0.0)
        axs[i].set_title('Power',x=0.07, y=0.03,ha="left")
    elif file == "util.parquet":
        axs[i].set_title('Utilization',x=0.07, y=0.03,ha="left")
        axs[i].legend(loc='lower right',frameon=True)
        axs[i].get_legend().get_frame().set_linewidth(0.0)
    elif file == "cooling_model.parquet":
        if c_cnt == 0:
            axs[i].set_title('PUE',x=0.05, y=0.8,ha="left")
            axs[i].legend(loc='upper right')
        elif c_cnt == 1:
            axs[i].set_title('Cooling Tower\nReturn\nTemperature',x=0.05, y=0.5,ha="left")
            axs[i].legend(loc='upper right')
        else:
            axs[i].set_title('Cooling Tower Supply Temperature',x=0.1, y=0.8)
            axs[i].legend(loc='upper right')
        c_cnt = c_cnt+1
    elif file == "loss_history.parquet":
        axs[i].set_title('Loss')
        axs[i].legend(loc='upper right')
    else:
        raise KeyError()
#plt.show()
#plt.savefig(f"3in1.png",bbox_inches='tight')
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.subplots_adjust(hspace=0)
plt.tight_layout(pad=0,w_pad=0.0,h_pad=-0.08)#3)
plt.savefig(f"2in1-pm100day50.png",bbox_inches='tight',pad_inches = 0.02, dpi = 300)

