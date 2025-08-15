import networkx as nx
from itertools import combinations
from raps.utils import get_current_utilization


class NetworkModel:
    """
    """

    def __init__(self, *, available_nodes, config, **kwargs):
        self.topology = config.get('TOPOLOGY')
        # if fat-tree, build the graph once
        if self.topology == "fat-tree":
            print("building fat-tree graph...")
            self.fattree_k = config.get("FATTREE_K")
            self.net_graph = build_fattree(self.fattree_k)
            print(self.net_graph)
        elif self.topology == "dragonfly":
            print("building dragonfly graph...")
            D = config["DRAGONFLY_D"]     # groups
            A = config["DRAGONFLY_A"]     # routers per group
            P = config["DRAGONFLY_P"]     # hosts per router
            self.net_graph = build_dragonfly(D, A, P)
            print(self.net_graph)

            real_ids = available_nodes
            real_ids.sort()
            self.real_to_fat_idx = {rid: idx for idx, rid in enumerate(real_ids)}
            # e.g. real_to_fat_idx[10] = 0, real_to_fat_idx[11] = 1, etc., up to 791 → 791
        self.max_link_bw = config.get("NETWORK_MAX_BW")

    def simulate_network_utilization(self, *, job, debug=False):
        net_util = 0
        net_cong = 0
        net_tx = 0
        net_rx = 0
        max_throughput = self.max_link_bw * job.trace_quanta  # self.config.get('TRACE_QUANTA')  # Why? What should this be?

        if job.nodes_required <= 1:
            # single node, no network utilization or congestion.
            pass
        else:

            net_tx = get_current_utilization(job.ntx_trace, job)  # Are these % or actual bytes?
            net_rx = get_current_utilization(job.nrx_trace, job)
            net_util = network_utilization(net_tx, net_rx, max_throughput)

            # Congestion depends on topology:
            if self.topology == "fat-tree":
                # Map integers to hostnames
                host_list = [node_id_to_host_name(n, self.fattree_k) for n in job.scheduled_nodes]
                loads = link_loads_for_job(self.net_graph, host_list, net_tx)  # ? Only tx not rx or total net_util)
                net_cong = worst_link_util(loads, max_throughput)

                if debug:
                    print("  fat-tree hosts:", host_list)

            elif self.topology == "dragonfly":
                D = self.config["DRAGONFLY_D"]
                A = self.config["DRAGONFLY_A"]
                P = self.config["DRAGONFLY_P"]

                host_list = []
                for real_n in job.scheduled_nodes:
                    fat_idx = self.real_to_fat_idx[real_n]   # contiguous in [0..(D*A*P−1)]
                    host_list.append(dragonfly_node_id_to_host_name(fat_idx, D, A, P))
                if debug:
                    print("  dragonfly hosts:", host_list)
                ##if len(host_list) <= 1:
                #    net_cong = 0.0
                #else:
                loads = link_loads_for_job(self.net_graph, host_list, net_tx)  # ? Only tx not rx or total net_util)
                net_cong = worst_link_util(loads, max_throughput)

            else:  # capacity model: simple α+β or normalized overload
                net_cong = network_congestion(net_tx, net_rx, max_throughput)

        return net_util, net_cong, net_tx, net_rx, max_throughput


def apply_job_slowdown(*,job, max_throughput, net_util, net_cong, net_tx, net_rx, debug: bool = False):
    # Get the maximum allowed bandwidth from the configuration.
    if net_cong > 1:
        if debug:
            print(f"congested net_cong: {net_cong}, max_throughput: {max_throughput}")
            print(f"length of {len(job.gpu_trace)} before dilation")
        throughput = net_tx + net_rx
        slowdown_factor = network_slowdown(throughput, max_throughput)

        if debug:
            print("***", hasattr(job, 'dilated'), throughput, max_throughput, slowdown_factor)

        # Only apply slowdown once per job to avoid compounding the effect.
        if not job.dilated:
            if debug:
                print(f"Applying slowdown factor {slowdown_factor:.2f} to job {job.id} due to network congestion")
            job.apply_dilation(slowdown_factor)
            job.dilated = True
            if debug:
                print(f"length of {len(job.gpu_trace)} after dilation")
    else:
        slowdown_factor = 1
    job.slowdown_factor = slowdown_factor

    return slowdown_factor


def compute_system_network_stats(net_utils,net_tx_list,net_rx_list,slowdown_factors):

    # Compute network averages
    n = len(net_utils) or 1
    avg_tx = sum(net_tx_list) / n
    avg_rx = sum(net_rx_list) / n
    avg_net = sum(net_utils) / n
    #avg_slowdown_per_job = sum(slowdown_factors) / n
    #self.avg_slowdown_history.append(avg_slowdown_per_job)
    #max_slowdown_per_job = max(slowdown_factors)
    #self.max_slowdown_history.append(max_slowdown_per_job)

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
        if len(job_hosts) >= 2:
            per_peer = tx_volume_bytes / (len(job_hosts)-1)
        else:
            per_peer = 0
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


def build_dragonfly(D: int, A: int, P: int) -> nx.Graph:
    """
    Build a “simple” k-ary Dragonfly with:
       D = # of groups
       A = # of routers per group
       P = # of hosts (endpoints) per router

    Naming convention:
      - Router nodes: "r_{g}_{r}"   with g ∈ [0..D−1], r ∈ [0..A−1]
      - Host  nodes: "h_{g}_{r}_{p}"  with p ∈ [0..P−1]

    Topology:
      1. All routers within a group form a full clique.
      2. Each router r in group g has exactly one “global link” to router r in each other group.
      3. Each router r in group g attaches to P hosts ("h_{g}_{r}_{0..P−1}").
    """
    G = nx.Graph()

    # 1) Create all router nodes
    for g in range(D):
        for r in range(A):
            router = f"r_{g}_{r}"
            G.add_node(router, type="router", group=g, index=r)

    # 2) Intra‐group full mesh of routers
    for g in range(D):
        routers_in_group = [f"r_{g}_{r}" for r in range(A)]
        for u, v in combinations(routers_in_group, 2):
            G.add_edge(u, v)

    # 3) Inter‐group “one‐to‐one” global links
    #    (router index r in group g  →  router index r in group g2)
    for g1 in range(D):
        for g2 in range(g1 + 1, D):
            for r in range(A):
                u = f"r_{g1}_{r}"
                v = f"r_{g2}_{r}"
                G.add_edge(u, v)

    # 4) Attach hosts to each router
    for g in range(D):
        for r in range(A):
            router = f"r_{g}_{r}"
            for p in range(P):
                host = f"h_{g}_{r}_{p}"
                G.add_node(host, type="host", group=g, router=r, index=p)
                G.add_edge(router, host)

    return G


def dragonfly_node_id_to_host_name(fat_idx: int, D: int, A: int, P: int) -> str:
    """
    Given a contiguous fat‐index ∈ [0..(D*A*P − 1)], return "h_{g}_{r}_{p}".
    Hosts are laid out in order:
      0..(P−1)    → group=0, router=0, p=0..P−1
      P..2P−1     → group=0, router=1, p=0..P−1
      …
      (A*P)..(2A*P−1) → group=1, router=0, …
    In general:
       host_offset      = fat_idx % P
       router_offset    = (fat_idx // P) % A
       group            = fat_idx // (A*P)
    """
    total_hosts = D * A * P
    assert 0 <= fat_idx < total_hosts, "fat_idx out of range"

    host_offset   = fat_idx % P
    router_group  = (fat_idx // P) % A
    pod           = fat_idx // (A * P)
    return f"h_{pod}_{router_group}_{host_offset}"
