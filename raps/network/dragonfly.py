import networkx as nx
from itertools import combinations


import networkx as nx

def build_dragonfly(d, a, p):
    """
    Build a Dragonfly network graph.
    d = routers per group
    a = global connections per router
    p = compute nodes per router
    """
    G = nx.Graph()
    num_groups = a + 1  # standard Dragonfly rule

    # --- Routers and hosts ---
    for g in range(num_groups):
        for r in range(d):
            router = f"r_{g}_{r}"
            G.add_node(router, layer="router", group=g)

            # attach p hosts to each router
            for h in range(p):
                host = f"h_{g}_{r}_{h}"
                G.add_node(host, layer="host", group=g)
                G.add_edge(router, host)

    # --- Intra-group full mesh ---
    for g in range(num_groups):
        routers = [f"r_{g}_{r}" for r in range(d)]
        for i in range(d):
            for j in range(i + 1, d):
                G.add_edge(routers[i], routers[j])

    # --- Inter-group (global) links ---
    for g in range(num_groups):
        for r in range(d):
            src = f"r_{g}_{r}"
            for offset in range(1, a + 1):
                dst_group = (g + offset) % num_groups
                dst = f"r_{dst_group}_{r % d}"
                G.add_edge(src, dst)

    return G


def build_dragonfly2(D: int, A: int, P: int) -> nx.Graph:
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

    Examples
    --------
    >>> from raps.plotting import plot_network_graph
    >>> G = build_dragonfly(D=2, A=2, P=2)
    >>> plot_network_graph(G, 'dragonfly.png')
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
    Convert a contiguous Dragonfly host index to its hierarchical name.

    For a Dragonfly with:
      D routers per group,
      A global links per router  ⇒ num_groups = A + 1,
      P compute nodes per router.

    Hosts are laid out in contiguous order:
      group g = floor(fat_idx / (D * P))
      router r = (fat_idx // P) % D
      host   h = fat_idx % P
    """
    num_groups = A + 1
    total_hosts = num_groups * D * P
    assert 0 <= fat_idx < total_hosts, f"fat_idx {fat_idx} out of range (max {total_hosts-1})"

    group = fat_idx // (D * P)
    router = (fat_idx // P) % D
    host = fat_idx % P
    return f"h_{group}_{router}_{host}"


def build_dragonfly_idx_map(d: int, a: int, p: int, total_real_nodes: int) -> dict[int, str]:
    """
    Build a mapping {real_node_index: host_name} for Dragonfly.
    Wrap around if total_real_nodes > total_hosts.
    """
    num_groups = a + 1
    total_hosts = num_groups * d * p

    mapping = {}
    for i in range(total_real_nodes):
        fat_idx = i % total_hosts  # <- wrap safely
        group = fat_idx // (d * p)
        router = (fat_idx // p) % d
        host = fat_idx % p
        mapping[i] = f"h_{group}_{router}_{host}"
    return mapping
