import csv
import networkx as nx
from pathlib import Path


def build_torus3d(dims, wrap=True, link_bw=1e9, hosts_per_router=1, routing="DOR_XYZ", coords_csv=None):
    """
    Build a 3D torus at router granularity, then attach host nodes to routers.
    Node ids in the returned graph are host names ("h_x_y_z_i") and router names ("r_x_y_z").
    Edges have attribute 'capacity' (bytes/s) and 'latency' (per hop).
    """
    X, Y, Z = map(int, dims)
    G = nx.Graph()

    # Routers
    def rname(x, y, z):
        return f"r_{x}_{y}_{z}"

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                G.add_node(rname(x, y, z), kind="router", coord=(x, y, z))

    # Toroidal links between routers (±x, ±y, ±z)
    def wrapi(i, n):
        return (i + n) % n if wrap else (None if i < 0 or i >= n else i)

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                u = rname(x, y, z)
                # x+
                nxp = wrapi(x + 1, X)
                v = rname(nxp, y, z) if nxp is not None else None
                if v and not G.has_edge(u, v):
                    G.add_edge(u, v, capacity=link_bw)
                # y+
                nyp = wrapi(y + 1, Y)
                v = rname(x, nyp, z) if nyp is not None else None
                if v and not G.has_edge(u, v):
                    G.add_edge(u, v, capacity=link_bw)
                # z+
                nzp = wrapi(z + 1, Z)
                v = rname(x, y, nzp) if nzp is not None else None
                if v and not G.has_edge(u, v):
                    G.add_edge(u, v, capacity=link_bw)

    # Attach hosts to routers
    host_to_router = {}
    router_to_hosts = {}

    def hname(x, y, z, i):
        return f"h_{x}_{y}_{z}_{i}"

    # If a nid→(x,y,z) CSV is supplied, place accordingly; else dense round-robin
    # CSV format: nid,x,y,z[,i]
    nid_placement = {}
    if coords_csv:
        p = Path(coords_csv)
        with p.open("rt") as fh:
            rd = csv.reader(fh)
            for row in rd:
                if not row:
                    continue
                nid = int(row[0])
                x, y, z = map(int, row[1:4])
                i = int(row[4]) if len(row) > 4 else 0
                nid_placement[nid] = (x, y, z, i)

    # Build hosts
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                r = rname(x, y, z)
                router_to_hosts[r] = []
                for i in range(hosts_per_router):
                    h = hname(x, y, z, i)
                    G.add_node(h, kind="host", coord=(x, y, z), local_index=i)
                    G.add_edge(h, r, capacity=link_bw)  # host↔router edge; you can cap with NETWORK_MAX_BW instead
                    host_to_router[h] = r
                    router_to_hosts[r].append(h)

    meta = {
        "dims": (X, Y, Z),
        "wrap": wrap,
        "routing": routing,
        "host_to_router": host_to_router,
        "router_to_hosts": router_to_hosts,
    }
    return G, meta


def _axis_steps(a, b, n, wrap=True):
    """Return minimal step sequence along one axis from a to b with wrap-around."""
    if a == b:
        return []
    fwd = (b - a) % n
    back = (a - b) % n
    if not wrap:
        step = 1 if b > a else -1
        return [step] * abs(b - a)
    if fwd <= back:
        return [1] * fwd
    else:
        return [-1] * back


def torus_route_xyz(src_r, dst_r, dims, wrap=True):
    """Router-level path (list of router names) using XYZ dimension-order routing."""
    X, Y, Z = dims

    def parse(r):
        _, x, y, z = r.split("_")
        return int(x), int(y), int(z)

    x1, y1, z1 = parse(src_r)
    x2, y2, z2 = parse(dst_r)

    path = [src_r]
    x, y, z = x1, y1, z1
    for step in _axis_steps(x, x2, X, wrap):
        x = (x + step) % X
        path.append(f"r_{x}_{y}_{z}")
    for step in _axis_steps(y, y2, Y, wrap):
        y = (y + step) % Y
        path.append(f"r_{x}_{y}_{z}")
    for step in _axis_steps(z, z2, Z, wrap):
        z = (z + step) % Z
        path.append(f"r_{x}_{y}_{z}")
    return path


def torus_host_path(G, meta, h_src, h_dst):
    r_src = meta["host_to_router"][h_src]
    r_dst = meta["host_to_router"][h_dst]
    routers = torus_route_xyz(r_src, r_dst, meta["dims"], meta["wrap"])
    # host->src_router + (router path) + dst_router->host
    path = [h_src, r_src] + routers[1:] + [h_dst]
    return path


def link_loads_for_job_torus(G, meta, host_list, traffic_bytes):
    # all-to-all between hosts in host_list, route via torus_host_path, add traffic_bytes per pair
    loads = {}
    n = len(host_list)
    for i in range(n):
        for j in range(i + 1, n):
            p = torus_host_path(G, meta, host_list[i], host_list[j])
            for u, v in zip(p, p[1:]):
                e = tuple(sorted((u, v)))
                loads[e] = loads.get(e, 0) + traffic_bytes
    return loads
