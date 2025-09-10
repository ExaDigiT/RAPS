import networkx as nx
from itertools import combinations


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

    host_offset = fat_idx % P
    router_group = (fat_idx // P) % A
    pod = fat_idx // (A * P)
    return f"h_{pod}_{router_group}_{host_offset}"
