#!/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

style=['seaborn-v0_8', 'tableau-colorblind10']

for j in range(-1,len(style)):
    if j in range(0,len(style)):
        plt.style.use(style[j])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Revert to the default style
        plt.style.use('default')
        # Apply ggplot colors to default style
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} <simulation_result/dir>")
        exit()

    # e.g. path = "$HOME/Repositories/exadigit/raps/simulation_results/b803010"

    policies = ['fcfs-nobf','fcfs-easy','priority-nobf','priority-easy','priority-ffbf','replay']
    files = ['cooling_model.parquet', 'loss_history.parquet', 'power_history.parquet', 'util.parquet']
    files = ['cooling_model.parquet', 'power_history.parquet', 'util.parquet']
    files = ['util.parquet', 'power_history.parquet', 'cooling_model.parquet']
    #files = ['loss_history.parquet', 'power_history.parquet', 'util.parquet']
    #files = ['power_history.parquet', 'util.parquet', 'cooling_model.parquet']

    policy_path = {f"{policy}":f"{path}/{policy}" for policy in policies}
    full_files = {f"{policy}":f"{path}/{policy}/{file}" for policy in policies for file in files}


    def iter_to_seconds(i):
        return i * 15


    fig, axs = plt.subplots(len(files),figsize=(12, 12))
    for i,file in enumerate(files):
        policy_files = [f"{path}/{policy}/{file}" for policy in policies]
        for policy_file in policy_files:
            # df = pd.read_parquet(policy_file)
            x = 'time'
            policy = policy_file.split('/')[-2]
            if file == "power_history.parquet":
                y = 'power [kw]'
                ylab = 'Power [kW]'
                ylim = 29000
                axs[i].set_ylim(0,ylim)

                df = pd.read_parquet(policy_file)
                df = df.rename(columns={0:'time',1:'power [kw]'})

            elif file == "cooling_model.parquet":
                y = 'pue'
                ylab = 'PUE'

                df = pd.read_parquet(policy_file)
                df['index'] = df.index
                df[x] = df['index'].apply(iter_to_seconds)
                ymax = max(df['pue'])
                #axs[i].plot(df[x],df[y], label=ylab)

            elif file == "loss_history.parquet":
                y = 'loss [kw]'
                ylab = 'Loss [kW]'
                ylim = 29000
                axs[i].set_ylim(0,ylim)

                df = pd.read_parquet(policy_file)
                df = df.rename(columns={0:'time',1:'loss [kw]'})
                #axs[i].plot(df[x],df[y], label=ylab)

            elif file == "util.parquet":
                y = 'utilization'
                ylab = 'Utilization'

                df = pd.read_parquet(policy_file)
                df = df.rename(columns={0:'time', 1:'utilization [%]'})
                df[y] = df['utilization [%]'] / 100
                #axs[i].plot(df[x], df[y], label=ylab)

            else:
                raise KeyError

            axs[i].plot(df[x],df[y], label=policy)
            axs[i].set_ylabel(ylab)
            #$axs[i].plot(df[0],df[1],label=policy)
        if file == "power_history.parquet":
            axs[i].legend(loc='upper right')
            axs[i].set_title('Power')
        elif file == "util.parquet":
            axs[i].set_title('Utilization')
            axs[i].legend(loc='lower right')
        elif file == "cooling_model.parquet":
            axs[i].set_title('PUE')
            axs[i].legend(loc='upper right')
        elif file == "loss_history.parquet":
            axs[i].set_title('Loss')
            axs[i].legend(loc='upper right')
        else:
            raise KeyError()
    #plt.show()
    plt.savefig(f"Type{[j]}.png")


    #for i in [1]:
    #    fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    #    power = path + "/" + files[2]
    #    loss = path + "/" + files[1]
    #    util = path + "/" + files[3]
    #
    #    df_power = pd.read_parquet(power)
    #    df_power = df_power.rename(columns={0:'time',1:'power [kw]'})
    #    ax1.plot(df_power['time'],df_power['power [kw]'], color='black', label='Power kW]')
    #
    #    #df_loss = pd.read_parquet(loss)
    #    #df_loss = df_loss.rename(columns={0:'time',1:'loss [kw]'})
    #    #ax1.plot(df_loss['time'],df_loss['loss [kw]'], color='red', label='Loss [kW]')
    #
    #    ax2 = ax1.twinx()
    #
    #    #df_cooling = pd.read_parquet(cooling)
    #    #df_cooling['index'] = df_cooling.index
    #    #df_cooling['time'] = df_cooling['index'].apply(iter_to_seconds)
    #    #ymax = max(df_cooling['pue'])
    #    #ax2.plot(df_cooling['time'],df_cooling['pue'], color='blue', label='PUE')
    #
    #    df_util = pd.read_parquet(util)
    #    df_util = df_util.rename(columns={0:'time', 1:'utilization [%]'})
    #    df_util['utilization'] = df_util['utilization [%]'] / 100
    #    ax2.plot(df_util['time'],df_util['utilization'], color='orange', label='Utilization')
    #
    #    #ymax = max(max(df_cooling['pue']),max(df_util['utilization']))
    #    ymax = max(0,max(df_util['utilization']))
    #    ax2.set_ylim([0, ymax * 1.05])
    #
    #    ax1.set_xlabel('time [s]')
    #    ax1.set_ylabel('[kW]')
    #    ax2.set_ylabel('[%]')
    #    plt.title(path)
    #    ax1.legend(loc='upper left')
    #    ax2.legend(loc='upper right')
    #    plt.show()
    #    #plt.savefig("test.png")
