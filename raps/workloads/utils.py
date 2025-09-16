import math
import numpy as np
import matplotlib.pyplot as plt


def plot_job_hist(jobs, config=None, dist_split=None, gantt_nodes=False):
    # put args.multimodal in dist_split!
    split = [1.0]
    num_dist = 1
    if dist_split:
        num_dist = len(dist_split)
        split = dist_split

    y = [y.nodes_required for y in jobs]
    x = [x.expected_run_time for x in jobs]
    x2 = [x.time_limit for x in jobs]
    fig_m = plt.figure()
    gs = fig_m.add_gridspec(30, 1)
    gs0 = gs[0:20].subgridspec(500, 500, hspace=0, wspace=0)
    gs1 = gs[24:].subgridspec(1, 1)

    ax_top = fig_m.add_subplot(gs0[:])
    ax_top.axis('off')
    ax_top.set_title('Job Distribution')

    ax_bot = fig_m.add_subplot(gs1[:])
    ax_bot.axis('off')
    ax_bot.set_title('Submit Time + Wall Time')

    # ax0 = fig_m.add_subplot(gs[:2,:])
    # ax1 = fig_m.add_subplot(gs[2:,:])

    # gss = gridspec.GridSpec(5, 5, figure=ax0)
    # fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': (4, 1), 'height_ratios': (1, 4)})
    axs = []
    col = []
    col.append(fig_m.add_subplot(gs0[:100, :433]))
    col.append(fig_m.add_subplot(gs0[:100, 433:]))
    axs.append(col.copy())
    col = []
    col.append(fig_m.add_subplot(gs0[100:, :433]))
    col.append(fig_m.add_subplot(gs0[100:, 433:]))
    axs.append(col.copy())

    ax_b = fig_m.add_subplot(gs1[:, :])

    # Create scatter plot
    for i in range(len(x)):
        axs[1][0].plot([x[i], x2[i]], [y[i], y[i]], color='lightblue', zorder=1)
    axs[1][0].scatter(x2, y, marker='.', c='lightblue', zorder=2)
    axs[1][0].scatter(x, y, zorder=3)

    cpu_util = [x.cpu_trace for x in jobs]
    if isinstance(cpu_util[0], np.ndarray):
        cpu_util = np.concatenate(cpu_util).ravel()
    elif isinstance(cpu_util[0], list):
        cpu_util = [sum(part) / len(part) for part in cpu_util]
    gpu_util = [x.gpu_trace for x in jobs]
    if isinstance(gpu_util[0], np.ndarray):
        gpu_util = np.concatenate(gpu_util).ravel()
    elif isinstance(gpu_util[0], list):
        gpu_util = [sum(part) / len(part) for part in gpu_util]
    if not all([x == 0 for x in gpu_util]):
        axs[0][1].scatter(cpu_util, gpu_util, zorder=2, marker='.', s=0.2)
        axs[0][1].hist(gpu_util, bins=100, orientation='horizontal', zorder=1, density=True, color='tab:purple')
        axs[0][1].axhline(np.mean(gpu_util), color='r', linewidth=1, zorder=3)
        axs[0][1].set(ylim=[0, config['GPUS_PER_NODE']])
        axs[0][1].set_ylabel("gpu util")
        axs[0][1].yaxis.set_label_coords(1.15, 0.5)
        axs[0][1].yaxis.set_label_position("right")
        axs[0][1].yaxis.tick_right()
    else:
        axs[0][1].set_yticks([])
    axs[0][1].hist(cpu_util, bins=100, orientation='vertical', zorder=1, density=True, color='tab:cyan')
    axs[0][1].axvline(np.mean(cpu_util), color='r', linewidth=1, zorder=3)
    axs[0][1].set(xlim=[0, config['CPUS_PER_NODE']])
    axs[0][1].set_xlabel("cpu util")
    axs[0][1].xaxis.set_label_coords(0.5, 1.30)
    axs[0][1].xaxis.set_label_position("top")
    axs[0][1].xaxis.tick_top()
    axs[0][0].hist(x2, bins=max(1, math.ceil(min(100, (max(x2) - min(x))))), orientation='vertical', color='lightblue')
    axs[0][0].hist(x, bins=max(1, math.ceil(min(100, (max(x2) - min(x))))), orientation='vertical')
    axs[1][0].sharex(axs[0][0])
    axs[1][1].hist(y, bins=max(1, min(100, (max(y) - min(y)))), orientation='horizontal')
    axs[1][0].sharey(axs[1][1])

    # Remove ticks
    axs[0][0].set_xticks([])
    axs[1][1].set_yticks([])
    axs[0][1].spines['top'].set_color('white')
    axs[0][1].spines['right'].set_color('white')
    axs[1][0].set_ylabel("nodes [N]")
    axs[1][0].set_xlabel("wall time [hh:mm]")
    minx_s = 0
    maxx_s = math.ceil(max(x2))
    x_label_mins = [n for n in np.arange(minx_s // 60, maxx_s // 60)]
    x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
                   (x1, x2) in [(n // 60, n % 60) for
                                n in x_label_mins[0::60]]]
    axs[1][0].set_xticks(x_label_ticks, x_label_str)
    miny = min(y)
    maxy = max(y)
    interval = max(1, maxy // 10)
    y_ticks = np.arange(0, maxy, interval)
    y_ticks[0] = miny
    axs[1][0].set_yticks(y_ticks)

    axs[0][0].tick_params(axis="x", labelbottom=False)
    axs[1][1].tick_params(axis="y", labelleft=False)

    # Submit_time and Wall_time
    duration = [x.expected_run_time for x in jobs]
    nodes_required = [x.nodes_required for x in jobs]
    submit_t = [x.submit_time for x in jobs]

    offset = 0
    split_index = 0
    split_offset = math.floor(len(x) * split[split_index])
    if gantt_nodes:
        if split[0] == 0.0:
            ax_b.axhline(y=offset, color='red', linestyle='--', lw=0.5)
            split_index += 1
        for i in range(len(x)):
            # ax_b.barh(i,duration[i], height=1.0, left=submit_t[i])
            ax_b.barh(offset + nodes_required[i] / 2, duration[i], height=nodes_required[i], left=submit_t[i])
            offset += nodes_required[i]
            if i != len(x) - 1 and i == split_offset - 1 and split_index < len(split):
                ax_b.axhline(y=offset, color='red', linestyle='--', lw=0.5)
                split_index += 1
                split_offset += math.floor(len(x) * split[split_index])
                # ax_b.axhline(y=(len(x)/num_dist * i)-0.5, color='red', linestyle='--',lw=0.5)
        if split[-1] == 0.0:
            ax_b.axhline(y=offset, color='red', linestyle='--', lw=0.5)
            split_index += 1
        ax_b.set_ylabel("Jobs' acc. nodes")
    else:
        for i in range(len(x)):
            ax_b.barh(i, duration[i], height=1.0, left=submit_t[i])
        for i in range(1, num_dist):
            if num_dist == 1:
                break
            ax_b.axhline(y=(len(x) * split[split_index]) - 0.5, color='red', linestyle='--', lw=0.5)
            split_index += 1
        ax_b.set_ylabel("Job ID")
        # ax_b labels:
    ax_b.set_xlabel("time [hh:mm]")
    minx_s = 0
    maxx_s = math.ceil(max([x.expected_run_time for x in jobs]) + max([x.submit_time for x in jobs]))
    x_label_mins = [n for n in np.arange(minx_s // 60, maxx_s // 60)]
    x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
                   (x1, x2) in [(n // 60, n % 60) for
                                n in x_label_mins[0::60]]]

    ax_b.set_xticks(x_label_ticks, x_label_str)
    ax_b.yaxis.set_inverted(True)

    plt.show()
