"""
This module provides functionality for creating and saving various types of plots using Matplotlib.

The module defines a `BasePlotter` class for setting up plots and saving them, and a `Plotter` class
that extends `BasePlotter` to include methods for plotting histories, histograms, and comparisons.

Classes
-------
BasePlotter
    A base class for setting up and saving plots.
Plotter
    A class for creating and saving specific types of plots, such as histories,
    histograms, and comparisons.
"""

import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

import time
import numpy as np
import networkx as nx
import random
from uncertainties import unumpy
from rich.progress import track


class BasePlotter:
    """
    A base class for setting up and saving plots.

    Attributes
    ----------
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The title of the plot.
    """

    def __init__(self, xlabel, ylabel, title, uncertainties=False):
        """
        Constructs all the necessary attributes for the BasePlotter object.

        Parameters
        ----------
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        title : str
            The title of the plot.
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.uncertainties = uncertainties

    def setup_plot(self, figsize=(10, 5)):
        """
        Sets up the plot with the given figure size, labels, title, and grid.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure (default is (10, 5)).
        """
        plt.figure(figsize=figsize)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.grid(True)

    def save_and_close_plot(self, save_path):
        """
        Saves the plot to the specified path and closes the plot.

        Parameters
        ----------
        save_path : str
            The path to save the plot.
        """
        plt.savefig(save_path)
        plt.close()


class Plotter(BasePlotter):
    """
    A class for creating and saving specific types of plots, such as histories,
    histograms, and comparisons.

    Attributes
    ----------
    save_path : str
        The path to save the plot.
    """

    def __init__(self, xlabel='', ylabel='', title='', save_path: Path | str = 'out.svg', uncertainties=False):
        """
        Constructs all the necessary attributes for the Plotter object.

        Parameters
        ----------
        xlabel : str, optional
            The label for the x-axis (default is an empty string).
        ylabel : str, optional
            The label for the y-axis (default is an empty string).
        title : str, optional
            The title of the plot (default is an empty string).
        save_path : str, optional
            The path to save the plot (default is 'out.svg').
        uncertainties: boolean, optional
            Flag if uncertainties are enabled and ufloats are used.
        """
        super().__init__(xlabel, ylabel, title, uncertainties)
        self.save_path = save_path

    def plot_history(self, x, y):
        """
        Plots a history plot of the given x and y values and saves it.

        Parameters
        ----------
        x : list
            The x values for the plot.
        y : list
            The y values for the plot.
        """
        self.setup_plot()

        if self.uncertainties:
            nominal_curve = plt.plot(x, unumpy.nominal_values(y))
            plt.fill_between(x, unumpy.nominal_values(y) - unumpy.std_devs(y),
                             unumpy.nominal_values(y) + unumpy.std_devs(y),
                             facecolor=nominal_curve[0].get_color(),
                             edgecolor='face', alpha=0.1, linewidth=0)
        else:
            plt.plot(x, y)
        self.save_and_close_plot(self.save_path)

    def plot_histogram(self, data, bins=50):
        """
        Plots a histogram of the given data and saves it.

        Parameters
        ----------
        data : list
            The data to plot in the histogram.
        bins : int, optional
            The number of bins in the histogram (default is 50).
        """
        self.setup_plot()
        plt.hist(data, bins=bins)
        self.save_and_close_plot(self.save_path)

    def plot_compare(self, x, y):
        """
        Plots a comparison plot of the given x and y values and saves it.

        Parameters
        ----------
        x : list
            The x values for the plot.
        y : list
            The y values for the plot.
        """
        self.setup_plot()
        plt.plot(x, y)
        self.save_and_close_plot(self.save_path)


def plot_nodes_histogram(nr_list, num_bins=25):
    print("plotting nodes required histogram...")

    # Create logarithmically spaced bins
    bins = np.logspace(np.log2(min(nr_list)), np.log2(max(nr_list)), num=num_bins, base=2)

    # Set up the figure
    plt.clf()
    plt.figure(figsize=(10, 3))

    # Create the histogram
    plt.hist(nr_list, bins=bins, edgecolor='black')

    # Add a title and labels
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')

    # Set the axes to logarithmic scale
    plt.xscale('log', base=2)
    plt.yscale('log')

    # Customize the x-ticks: Choose positions like 1, 8, 64, etc.
    ticks = [2**i for i in range(0, 14)]
    plt.xticks(ticks, labels=[str(tick) for tick in ticks])

    # Set min-max axis bounds
    plt.xlim(1, max(nr_list))

    # Save the histogram to a file
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')


def plot_submit_times(submit_times, nr_list):
    """Plot number of nodes over time"""

    print("plotting submit times...")

    # Determine the time scale
    max_time = max(submit_times)

    if max_time >= 3600 * 24 * 7:  # If more than a week convert time to days
        submit_times = [time / (3600 * 24) for time in submit_times]
        time_label = 'Submit Time (days)'
    elif max_time >= 3600 * 24:  # If more than 24 hours convert time to hours
        submit_times = [time / 3600 for time in submit_times]
        time_label = 'Submit Time (hours)'
    else:
        time_label = 'Submit Time (s)'

    plt.clf()
    plt.figure(figsize=(10, 2))

    # Create a bar chart
    bar_width = (max(submit_times) - min(submit_times)) / len(submit_times) * 0.8
    plt.bar(submit_times, nr_list, width=bar_width, color='blue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel(time_label)
    plt.ylabel('Number of Nodes')

    # Set min-max axis bounds
    plt.xlim(1, max(submit_times))

    # Set the y-axis to logarithmic scale with base 2
    plt.yscale('log', base=2)
    y_ticks = [2**i for i in range(0, 14)]
    plt.yticks(y_ticks, labels=[str(tick) for tick in y_ticks])

    # Save the plot to a file
    plt.savefig('submit_times.png', dpi=300, bbox_inches='tight')


def convert_time_scale(times):
    max_time = max(times)
    if max_time >= 3600 * 24 * 7:  # more than a week
        return [t / (3600 * 24) for t in times], 'days'
    elif max_time >= 3600 * 24:    # more than a day
        return [t / 3600 for t in times], 'hours'
    else:
        return times, 'seconds'


def plot_job_gantt(start_times, end_times, node_counts):
    # Convert times
    start_times, time_label = convert_time_scale(start_times)
    end_times, _ = convert_time_scale(end_times)

    plt.figure(figsize=(10, 4))

    # We'll plot each job in a different row on the Y-axis
    y_positions = range(len(start_times))  # 0, 1, 2, ...

    for s, e, n in zip(start_times, end_times, node_counts):
        # Bar placed at y = n
        plt.barh(
            y=n,                # node count is the vertical coordinate
            width=e - s,        # job duration on the x-axis
            left=s,             # start time
            height=0.8,         # thickness of the bar
            color='yellow',
            edgecolor='black',
            alpha=0.8
        )

    # for y, (s, e, n) in enumerate(zip(start_times, end_times, node_counts)):
    #    plt.barh(y, width=e - s, left=s, height=0.8,
    #                 color='yellow', edgecolor='black', alpha=0.8)
    #    # Optionally place the node count label in the middle of the bar
    #    plt.text((s + e)/2, y, str(n),
    #             ha='center', va='center', color='black')

    plt.xlabel(f'Time ({time_label})')
    plt.ylabel('Job Index')
    plt.title('Job Timeline (Gantt Style)')
    plt.yticks(y_positions)  # label each job if desired

    # Time axis from earliest start to latest end
    plt.xlim(min(start_times), max(end_times))

    plt.tight_layout()
    plt.savefig('job_gantt.png', dpi=300)


def plot_network_histogram(*, ax, data, bins=50, save_path='network_histogram.png'):
    """
    Plot a histogram of network traffic per job, with scientific notation on the x-axis.
    """
    if ax is None:
        ax = plt.figure(figsize=(10, 3))

    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)

    # log-scale the y-axis
    ax.yscale('log')

    # force scientific notation on x-axis
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    ax.xlabel('Network Traffic per Job (bytes)')
    ax.ylabel('Frequency')
    ax.title('Histogram of Network Traffic per Job')
    ax.grid(True, which='both', ls='--', lw=0.5)

    return ax


def spaced_colors(n, cmap_name='nipy_spectral'):
    cmap = plt.get_cmap(cmap_name)
    # Get n points spaced in [0,1]
    base = np.linspace(0, 1, n, endpoint=False)
    # Shuffle them to maximize distance between consecutive colors
    # e.g. take every k-th, wrap around
    step = int(np.ceil(np.sqrt(n)))
    indices = (step * np.arange(n)) % n
    values = base[indices]
    return [cmap(v) for v in values]


def plot_jobs_gantt(*, ax=None, jobs, bars_are_node_sized):
    jobs.sort(key=lambda x: x.submit_time)
    if ax is None:
        ax = plt.figure(figsize=(10, 4))
    # Submit_time and Wall_time
    submit_t = [x.submit_time for x in jobs]
    duration = [x.end_time - x.start_time if x.end_time and x.start_time else x.time_limit for x in jobs]
    nodes_required = [x.nodes_required for x in jobs]

    colors = spaced_colors(len(jobs))
    offset = 0
    for i in track(range(len(jobs)), description="Collecting information to plot"):
        if bars_are_node_sized:
            ax.barh(offset + nodes_required[i] / 2, duration[i], height=nodes_required[i], left=submit_t[i])
            offset += nodes_required[i]
        else:
            ax.barh(i, duration[i], height=1.0, left=submit_t[i], color=colors[i])
    print("Plotting")

    ax.set_ylabel("Job ID")
    # ax_b labels:
    ax.set_xlabel("time [hh:mm]")
    minx_s = min([x.submit_time for x in jobs])
    maxx_s = np.ceil(max([x.end_time - x.start_time if x.end_time and x.start_time else x.time_limit for
                          x in jobs]) + max([x.submit_time for x in jobs]))
    x_label_mins = [int(n) for n in np.arange(minx_s // 60, maxx_s // 60)]
    x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
                   (x1, x2) in [(n // 60, n % 60) for
                                n in x_label_mins[0::60]]]

    ax.set_xticks(x_label_ticks, x_label_str)
    # ax.yaxis.set_inverted(True)
    return ax


def plot_nodes_gantt(*, ax=None, jobs):
    if ax is None:
        ax = plt.figure(figsize=(10, 4))
    # Submit_time and Wall_time
    duration = [x.end_time - x.start_time if x.end_time and x.start_time else x.time_limit for x in jobs]
    # nodes_required = [x['nodes_required'] for x in jobs]
    start_t = [x.start_time for x in jobs]
    nodeIDs = [x.scheduled_nodes for x in jobs]
    print(nodeIDs)
    if not any(nodeIDs):
        raise IndexError(f"No nodeIDs: {nodeIDs}, jobs have no scheduled_nodes.")

    colors = spaced_colors(len(jobs))
    for i in track(range(len(jobs)), description="Collecting information to plot"):
        for nodeID in nodeIDs[i]:
            ax.barh(nodeID, duration[i], height=1.0, left=start_t[i], color=colors[i])
    print("Plotting")

    ax.set_ylabel("Node ID")
    # ax_b labels:
    ax.set_xlabel("time [hh:mm]")
    # minx_s = min([x.submit_time for x in jobs])  # Unused
    # maxx_s = np.ceil(max([x.wall_time for x in jobs]) + max([x.submit_time for x in jobs]))  # Unused
    # ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

    formatter = ticker.FuncFormatter(lambda s, x: time.strftime('%m-%d %H:%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # x_label_mins = [int(n) for n in np.arange(minx_s // 60, maxx_s // 60)]
    # x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    # x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
    #                        (x1,x2) in [(n // 60,n % 60) for
    #                                    n in x_label_mins[0::60]]]

    # ax.set_xticks(x_label_ticks,x_label_str)
    ax.set_ylim(1, max(list(itertools.chain.from_iterable(nodeIDs))))
    # ax.yaxis.set_inverted(True)
    return ax


def plot_fattree_hierarchy(G, k=32, save_path='net_fattree.png'):
    """Draw a hierarchical Fat-Tree layout with automatic scaling."""
    pos = {}

    # --- Layer order and matching prefixes ---
    layers = ["core", "agg", "edge", "h"]
    layer_prefixes = {
        "core": ["core", "c_"],
        "agg": ["agg", "a_"],
        "edge": ["edge", "e_"],
        "h": ["h", "host"]
    }

    # --- Compute how many nodes per layer ---
    layer_counts = {}
    for layer in layers:
        prefixes = layer_prefixes[layer]
        layer_nodes = [n for n in G.nodes if any(n.startswith(p) for p in prefixes)]
        layer_counts[layer] = len(layer_nodes)

    max_nodes = max(layer_counts.values()) or 1
    y_gap = 1.0 / (len(layers) - 1)

    # --- Assign positions, normalized to [0,1] range ---
    for j, layer in enumerate(layers):
        prefixes = layer_prefixes[layer]
        layer_nodes = [n for n in G.nodes if any(n.startswith(p) for p in prefixes)]
        n_layer = len(layer_nodes)
        if n_layer == 0:
            continue
        x_spacing = 1.0 / n_layer
        y = 1.0 - j * y_gap
        for i, node in enumerate(layer_nodes):
            x = (i + 0.5) * x_spacing  # center each node
            pos[node] = (x, y)

    # --- Draw figure ---
    plt.figure(figsize=(10, 8))
    color_map = {"core": "red", "agg": "orange", "edge": "green", "h": "blue"}
    size_map = {"core": 30, "agg": 20, "edge": 10, "h": 5}

    for layer in layers:
        nodes = [n for n in G.nodes if any(n.startswith(p) for p in layer_prefixes[layer])]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, node_color=color_map[layer],
                node_size=size_map[layer], label=layer.capitalize(), alpha=0.7
            )

    # --- Only draw inter-layer edges for clarity ---
    edgelist = [
        (u, v) for (u, v) in G.edges
        if not any(u.startswith(p) and v.startswith(p)
                   for p in ["c_", "a_", "e_", "h", "core", "agg", "edge", "host"])
    ]
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, alpha=0.05, width=0.4)

    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_dragonfly(G, save_path='net_dragonfly.png'):
    """
    Draw a circular Dragonfly layout: groups in a large ring,
    routers in small inner rings, hosts hanging around each router.
    """
    import math
    import matplotlib.pyplot as plt
    import networkx as nx

    # Identify groups
    groups = sorted({G.nodes[n]["group"] for n in G if "group" in G.nodes[n]})
    num_groups = len(groups)

    pos = {}
    R_outer = 1.0      # radius of the outer ring (groups)
    R_inner = 0.15     # radius of each group's internal ring

    # --- compute positions ---
    for i, g in enumerate(groups):
        # center of this group
        theta_g = 2 * math.pi * i / num_groups
        cx = R_outer * math.cos(theta_g)
        cy = R_outer * math.sin(theta_g)

        routers = [n for n in G if n.startswith("r_") and G.nodes[n]["group"] == g]
        hosts = [n for n in G if n.startswith("h_") and G.nodes[n]["group"] == g]

        # routers in small ring
        for j, r in enumerate(routers):
            theta_r = 2 * math.pi * j / len(routers)
            x = cx + R_inner * math.cos(theta_r)
            y = cy + R_inner * math.sin(theta_r)
            pos[r] = (x, y)

        # hosts slightly further out around each router
        for j, h in enumerate(hosts):
            router = f"r_{g}_{j // 8}" if len(routers) > 0 else None
            # angle toward router’s position if available
            angle = 2 * math.pi * (j / len(hosts))
            r_off = R_inner + 0.05
            x = cx + r_off * math.cos(angle)
            y = cy + r_off * math.sin(angle)
            pos[h] = (x, y)

    # --- Draw figure ---
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[n for n in G if n.startswith("r_")],
                           node_color="orange", node_size=20, label="Routers", alpha=0.9)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[n for n in G if n.startswith("h_")],
                           node_color="blue", node_size=8, label="Hosts", alpha=0.7)

    # intra-group edges light gray, inter-group black
    intra = [(u, v) for (u, v) in G.edges if G.nodes[u]["group"] == G.nodes[v]["group"]]
    inter = [(u, v) for (u, v) in G.edges if G.nodes[u]["group"] != G.nodes[v]["group"]]
    nx.draw_networkx_edges(G, pos, edgelist=intra, alpha=0.1, width=0.3, edge_color="gray")
    nx.draw_networkx_edges(G, pos, edgelist=inter, alpha=0.4, width=0.4, edge_color="black")

    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_torus2d(G, save_path="net_torus2d.png"):
    import matplotlib.pyplot as plt

    routers = [n for n, d in G.nodes(data=True) if d["type"] == "router"]
    hosts = [n for n, d in G.nodes(data=True) if d["type"] == "host"]

    fig, ax = plt.subplots(figsize=(8, 8))

    for u, v, d in G.edges(data=True):
        if d.get("type") == "router_link":
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.1, linewidth=0.5)

    # flatten z by adding it to y or x offset
    xs = [G.nodes[n]["x"] for n in routers]
    ys = [G.nodes[n]["y"] + 0.05 * G.nodes[n]["z"] for n in routers]
    ax.scatter(xs, ys, c="orange", s=10, label="Routers", alpha=0.8)

    hx = [G.nodes[n]["x"] for n in hosts]
    hy = [G.nodes[n]["y"] + 0.05 * G.nodes[n]["z"] for n in hosts]
    ax.scatter(hx, hy, c="blue", s=4, label="Hosts", alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y + (scaled Z)")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_torus3d(G, active_edges=None, max_edges=4000, save_path="net_torus3d.png"):
    """
    Plot a 3D torus with routers, hosts, and optional job link highlights.
    Args:
        G : networkx.Graph
        active_edges : list of (u,v) tuples for job links to highlight
        max_edges : subsample edges to avoid clutter
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Separate routers and hosts ---
    routers = [n for n, d in G.nodes(data=True) if d["type"] == "router"]
    hosts = [n for n, d in G.nodes(data=True) if d["type"] == "host"]

    # --- Plot routers ---
    xs, ys, zs = [G.nodes[n]["x"] for n in routers], [G.nodes[n]["y"]
                                                      for n in routers], [G.nodes[n]["z"] for n in routers]
    ax.scatter(xs, ys, zs, c="orange", s=6, label="Routers", alpha=0.8)

    # --- Plot hosts ---
    hx, hy, hz = [G.nodes[n]["x"] for n in hosts], [G.nodes[n]["y"] for n in hosts], [G.nodes[n]["z"] for n in hosts]
    ax.scatter(hx, hy, hz, c="dodgerblue", s=3, label="Hosts", alpha=0.6)

    # --- Draw router-to-router edges (subsampled) ---
    all_router_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "router_link"]
    if len(all_router_edges) > max_edges:
        all_router_edges = random.sample(all_router_edges, max_edges)
    for u, v in all_router_edges:
        x1, y1, z1 = G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u]["z"]
        x2, y2, z2 = G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v]["z"]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", alpha=0.05, linewidth=0.5)

    # --- Draw host links lightly ---
    for u, v, d in G.edges(data=True):
        if d.get("type") == "host_link":
            x1, y1, z1 = G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u]["z"]
            x2, y2, z2 = G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v]["z"]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color="lightblue", alpha=0.05, linewidth=0.3)

    # --- Overlay active job edges ---
    if active_edges:
        for u, v in active_edges:
            if u in G.nodes and v in G.nodes:
                x1, y1, z1 = G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u]["z"]
                x2, y2, z2 = G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v]["z"]
                ax.plot([x1, x2], [y1, y2], [z1, z2], color="red", linewidth=1.8, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    plotter = Plotter()
    # plotter.plot_history([1, 2, 3, 4])
