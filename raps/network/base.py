import networkx as nx


def debug_print_trace(job, label: str = ""):
    """Print either the length (if iterable) or the value of job.gpu_trace."""
    if hasattr(job.gpu_trace, "__len__"):
        print(f"length of {len(job.gpu_trace)} {label}")
    else:
        print(f"gpu_trace value {job.gpu_trace} {label}")


def apply_job_slowdown(*, job, max_throughput, net_util, net_cong, net_tx, net_rx, debug: bool = False):
    # Get the maximum allowed bandwidth from the configuration.
    if net_cong > 1:
        if debug:
            print(f"congested net_cong: {net_cong}, max_throughput: {max_throughput}")
            debug_print_trace(job, "before dilation")

        throughput = net_tx + net_rx
        slowdown_factor = network_slowdown(throughput, max_throughput)

        if debug:
            print("***", hasattr(job, "dilated"), throughput, max_throughput, slowdown_factor)

        # Only apply slowdown once per job to avoid compounding the effect.
        if not job.dilated:
            if debug:
                print(f"Applying slowdown factor {slowdown_factor:.2f} to job {job.id} due to network congestion")
            job.apply_dilation(slowdown_factor)
            job.dilated = True
            if debug:
                debug_print_trace(job, "after dilation")
    else:
        slowdown_factor = 1
    job.slowdown_factor = slowdown_factor

    return slowdown_factor


def compute_system_network_stats(net_utils, net_tx_list, net_rx_list, slowdown_factors):

    # Compute network averages
    n = len(net_utils) or 1
    avg_tx = sum(net_tx_list) / n
    avg_rx = sum(net_rx_list) / n
    avg_net = sum(net_utils) / n
    # avg_slowdown_per_job = sum(slowdown_factors) / n
    # self.avg_slowdown_history.append(avg_slowdown_per_job)
    # max_slowdown_per_job = max(slowdown_factors)
    # self.max_slowdown_history.append(max_slowdown_per_job)

    return avg_tx, avg_rx, avg_net


def network_congestion(tx, rx, max_throughput):
    """
    Overload factor ≥0: average of send/recv NOT clamped.
    >1.0 means you’re pushing above capacity.
    """
    tx_util = float(tx) / max_throughput
    rx_util = float(rx) / max_throughput
    return (tx_util + rx_util) / 2.0


def network_utilization(tx, rx, max_throughput):
    """
    True utilization in [0,1]: average of send/recv clamped to 100%.
    """
    tx_u = min(float(tx) / max_throughput, 1.0)
    rx_u = min(float(rx) / max_throughput, 1.0)
    return (tx_u + rx_u) / 2.0


def network_slowdown(current_throughput, max_throughput):
    """
    Calculate a slowdown factor based on current network bandwidth usage.

    If current_bw is within limits, the factor is 1.0 (no slowdown).
    If current_bw exceeds max_bw, the factor is current_bw/max_bw.
    """
    if current_throughput <= max_throughput:
        return 1.0
    else:
        return current_throughput / max_throughput


def all_to_all_paths(G, hosts):
    """
    Given a list of host names, return shortest‐paths for every unordered pair.
    """
    paths = []
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            src, dst = hosts[i], hosts[j]
            p = nx.shortest_path(G, src, dst)
            paths.append((src, dst, p))
    return paths


def link_loads_for_job(G, job_hosts, tx_volume_bytes):
    """
    Distribute tx_volume_bytes from each host equally to all its peers;
    accumulate per-link loads and return a dict {(u,v):bytes, …}.
    """
    paths = all_to_all_paths(G, job_hosts)
    loads = {edge: 0.0 for edge in G.edges()}
    # each host sends tx_volume_bytes to each of the (N-1) peers
    for src in job_hosts:
        if len(job_hosts) >= 2:
            per_peer = tx_volume_bytes / (len(job_hosts) - 1)
        else:
            per_peer = 0
        # find paths where src is the sender
        for s, d, p in paths:
            if s != src:
                continue
            # add per_peer to every link on p
            for u, v in zip(p, p[1:]):
                # ensure ordering matches loads keys
                edge = (u, v) if (u, v) in loads else (v, u)
                loads[edge] += per_peer
    return loads


def worst_link_util(loads, throughput):
    """
    Given loads in **bytes** and capacity in **bits/sec**, convert:
      util = (bytes * 8) / throughput
    Return the maximum util over all links.
    """
    max_util = 0.0
    for edge, byte_load in loads.items():
        util = (byte_load * 8) / throughput
        if util > max_util:
            max_util = util
    return max_util
