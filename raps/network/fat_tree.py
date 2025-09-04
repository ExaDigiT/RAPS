import networkx as nx


def node_id_to_host_name(node_id: int, k: int) -> str:
    """
    Convert an integer node id to the host name string in the fat-tree.
    Node IDs are assumed to be contiguous, mapping to h_{pod}_{edge}_{i}.
    """
    # need to match the scheme from build_fattree
    pod = node_id // (k * k // 4)
    edge = (node_id % (k * k // 4)) // (k // 2)
    host = node_id % (k // 2)
    return f"h_{pod}_{edge}_{host}"


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
    # num_core = (k//2)**2  # Unused!
    for i in range(k // 2):
        for j in range(k // 2):
            core = f"c_{i}_{j}"
            G.add_node(core, type="core")
    # pods
    for pod in range(k):
        # agg switches
        for agg in range(k // 2):
            a = f"a_{pod}_{agg}"
            G.add_node(a, type="agg")
            # connect to all core switches in column agg
            for i in range(k // 2):
                core = f"c_{agg}_{i}"
                G.add_edge(a, core)
        # edge switches + hosts
        for edge in range(k // 2):
            e = f"e_{pod}_{edge}"
            G.add_node(e, type="edge")
            # connect edge→each agg in this pod
            for agg in range(k // 2):
                a = f"a_{pod}_{agg}"
                G.add_edge(e, a)
            # connect hosts
            for h in range(k // 2):
                host = f"h_{pod}_{edge}_{h}"
                G.add_node(host, type="host")
                G.add_edge(e, host)
    return G
