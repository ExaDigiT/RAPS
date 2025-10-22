import pytest
from raps.network.dragonfly import build_dragonfly, dragonfly_node_id_to_host_name

def test_build_dragonfly():
    """Test building a small dragonfly network."""
    D, A, P = 2, 2, 2
    G = build_dragonfly(D, A, P)

    # Check number of nodes
    num_routers = D * A
    num_hosts = D * A * P
    total_nodes = num_routers + num_hosts
    assert len(G.nodes) == total_nodes

    # Check number of edges
    # Intra-group edges (clique)
    intra_group_edges = D * (A * (A - 1) // 2)
    # Inter-group edges
    inter_group_edges = A * (D * (D - 1) // 2)
    # Host to router edges
    host_router_edges = num_hosts
    total_edges = intra_group_edges + inter_group_edges + host_router_edges
    assert len(G.edges) == total_edges

    # Check node types
    node_types = [data["type"] for _, data in G.nodes(data=True)]
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
