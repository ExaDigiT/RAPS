from raps.network.dragonfly import build_dragonfly, dragonfly_node_id_to_host_name


def test_build_dragonfly():
    """Test building a small dragonfly network."""
    D = 2  # Routers per group
    A = 2  # Gloobal connections per router
    P = 2  # Compute nodes per router
    G = build_dragonfly(D, A, P)

    # Check number of nodes
    num_routers = D * (A + 1)
    num_hosts = num_routers * P
    total_nodes = num_routers + num_hosts
    assert len(G.nodes) == total_nodes

    # Check number of edges
    routers_per_group = D
    # Edges of the router clique:
    router_clique_edges_per_group = ((routers_per_group * (routers_per_group - 1)) // 2)
    # Edges for all router compute nodes:
    compute_node_edges_per_router = P
    # Total Intra-group edges:
    intra_group_edges = router_clique_edges_per_group + compute_node_edges_per_router * D

    # Inter-group edges
    total_groups = A + 1
    inter_group_edges_simple_clique = ((total_groups * (total_groups-1)) // 2)
    inter_group_edges = inter_group_edges_simple_clique * D
    # Host to router edges
    total_edges = intra_group_edges * total_groups + inter_group_edges
    assert len(G.edges) == total_edges

    # Check node types
    node_types = [data["layer"] for _, data in G.nodes(data=True)]
    assert node_types.count("router") == num_routers
    assert node_types.count("host") == num_hosts


def test_dragonfly_node_id_to_host_name():
    """Test the dragonfly_node_id_to_host_name function."""
    D, A, P = 2, 2, 2
    # Test a few node IDs
    assert dragonfly_node_id_to_host_name(0, D, A, P) == "h_0_0_0"
    assert dragonfly_node_id_to_host_name(1, D, A, P) == "h_0_0_1"
    assert dragonfly_node_id_to_host_name(2, D, A, P) == "h_0_1_0"
    assert dragonfly_node_id_to_host_name(3, D, A, P) == "h_0_1_1"
    assert dragonfly_node_id_to_host_name(4, D, A, P) == "h_1_0_0"
