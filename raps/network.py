import networkx as nx

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

def build_fattree(k):
    """
    Build a k-ary fat-tree:
      - k pods
      - each pod has k/2 edge switches, k/2 agg switches
      - core layer has (k/2)^2 core switches
      - each edge switch connects to k/2 hosts
    Returns a NetworkX Graph where:
      - hosts are named "h_{pod}_{edge}_{i}"
      - edge switches "e_{pod}_{edge}"
      - agg   switches "a_{pod}_{agg}"
      - core  switches "c_{i}_{j}"
    """
    G = nx.Graph()
    # core
    num_core = (k//2)**2
    for i in range(k//2):
        for j in range(k//2):
            core = f"c_{i}_{j}"
            G.add_node(core, type="core")
    # pods
    for pod in range(k):
        # agg switches
        for agg in range(k//2):
            a = f"a_{pod}_{agg}"
            G.add_node(a, type="agg")
            # connect to all core switches in column agg
            for i in range(k//2):
                core = f"c_{agg}_{i}"
                G.add_edge(a, core)
        # edge switches + hosts
        for edge in range(k//2):
            e = f"e_{pod}_{edge}"
            G.add_node(e, type="edge")
            # connect edge→each agg in this pod
            for agg in range(k//2):
                a = f"a_{pod}_{agg}"
                G.add_edge(e, a)
            # connect hosts
            for h in range(k//2):
                host = f"h_{pod}_{edge}_{h}"
                G.add_node(host, type="host")
                G.add_edge(e, host)
    return G

def all_to_all_paths(G, hosts):
    """
    Given a list of host names, return shortest‐paths for every unordered pair.
    """
    paths = []
    for i in range(len(hosts)):
        for j in range(i+1, len(hosts)):
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
        per_peer = tx_volume_bytes / (len(job_hosts)-1)
        # find paths where src is the sender
        for (s, d, p) in paths:
            if s != src: continue
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

def node_id_to_host_name(node_id: int, k: int) -> str:
    """
    Map a 0-based integer node_id into one of the fat-tree hosts "h_{pod}_{edge}_{h}".
    There are (k^3/4) total hosts, assigned in ascending order across pod → edge → h.
    """
    hosts_per_pod = (k // 2) * (k // 2)   # e.g. for k=8, hosts_per_pod = 16
    pod    = node_id // hosts_per_pod
    offset = node_id %  hosts_per_pod
    edge   = offset // (k // 2)
    idx    = offset %  (k // 2)
    return f"h_{pod}_{edge}_{idx}"
