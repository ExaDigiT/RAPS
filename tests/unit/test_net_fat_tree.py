import pytest
from raps.network.fat_tree import build_fattree, node_id_to_host_name

def test_build_fattree_k4():
    """Test building a k=4 fat-tree."""
    k = 4
    G = build_fattree(k, 16)

    # Check number of nodes
    num_hosts = k * (k // 2) * (k // 2)
    num_edge_switches = k * (k // 2)
    num_agg_switches = k * (k // 2)
    num_core_switches = (k // 2) ** 2
    total_nodes = num_hosts + num_edge_switches + num_agg_switches + num_core_switches
    assert len(G.nodes) == total_nodes

    # Check number of edges
    # Host to edge switch edges
    host_edges = num_hosts
    # Edge to agg switch edges
    edge_agg_edges = k * (k // 2) * (k // 2)
    # Agg to core switch edges
    agg_core_edges = k * (k // 2) * (k // 2)
    total_edges = host_edges + edge_agg_edges + agg_core_edges
    assert len(G.edges) == total_edges

    # Check node types
    node_types = [data["type"] for _, data in G.nodes(data=True)]
    assert node_types.count("host") == num_hosts
    assert node_types.count("edge") == num_edge_switches
    assert node_types.count("agg") == num_agg_switches
    assert node_types.count("core") == num_core_switches

def test_node_id_to_host_name():
    """Test the node_id_to_host_name function."""
    k = 4
    # Test a few node IDs
    assert node_id_to_host_name(0, k) == "h_0_0_0"
    assert node_id_to_host_name(1, k) == "h_0_0_1"
    assert node_id_to_host_name(2, k) == "h_0_1_0"
    assert node_id_to_host_name(3, k) == "h_0_1_1"
    assert node_id_to_host_name(4, k) == "h_1_0_0"
