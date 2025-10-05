import pytest
from raps.network.torus3d import build_torus3d, torus_route_xyz

def test_build_torus3d():
    """Test building a small 3D torus network."""
    dims = (2, 2, 2)
    G, meta = build_torus3d(dims)

    # Check number of nodes
    num_routers = dims[0] * dims[1] * dims[2]
    num_hosts = num_routers # hosts_per_router=1
    total_nodes = num_routers + num_hosts
    assert len(G.nodes) == total_nodes

    # Check number of edges
    # Router to router edges
    router_edges = (num_routers * 3) // 2 # Each router has 3 neighbors in a 3D torus
    # Host to router edges
    host_router_edges = num_hosts
    total_edges = router_edges + host_router_edges
    assert len(G.edges) == total_edges

    # Check node types
    node_types = [data["kind"] for _, data in G.nodes(data=True)]
    assert node_types.count("router") == num_routers
    assert node_types.count("host") == num_hosts

def test_torus_route_xyz():
    """Test the torus_route_xyz function."""
    dims = (4, 4, 4)
    # Test a simple route
    path = torus_route_xyz("r_0_0_0", "r_1_1_1", dims)
    assert path == ["r_0_0_0", "r_1_0_0", "r_1_1_0", "r_1_1_1"]

    # Test a route with wrap-around
    path = torus_route_xyz("r_3_3_3", "r_0_0_0", dims, wrap=True)
    assert path == ["r_3_3_3", "r_0_3_3", "r_0_0_3", "r_0_0_0"]

    # Test a route without wrap-around
    path = torus_route_xyz("r_0_0_0", "r_1_1_1", dims, wrap=False)
    assert path == ["r_0_0_0", "r_1_0_0", "r_1_1_0", "r_1_1_1"]
