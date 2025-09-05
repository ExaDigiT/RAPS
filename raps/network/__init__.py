from .base import (
    all_to_all_paths,
    apply_job_slowdown,
    compute_system_network_stats,
    link_loads_for_job,
    network_congestion,
    network_slowdown,
    network_utilization,
    worst_link_util,
)

from .fat_tree import build_fattree, node_id_to_host_name
from .torus3d import build_torus3d, link_loads_for_job_torus
from .dragonfly import build_dragonfly, dragonfly_node_id_to_host_name
from raps.utils import get_current_utilization

__all__ = [
    "NetworkModel",
    "apply_job_slowdown",
    "compute_system_network_stats",
    "network_congestion",
    "network_utilization",
    "network_slowdown",
    "all_to_all_paths",
    "link_loads_for_job",
    "worst_link_util",
    "build_fattree",
    "build_torus3d",
    "build_dragonfly",
    "dragonfly_node_id_to_host_name",
]


class NetworkModel:
    def __init__(self, *, available_nodes, config, **kwargs):
        self.config = config
        self.topology = config.get("TOPOLOGY")
        self.max_link_bw = config.get("NETWORK_MAX_BW", 1e9)  # default safeguard
        self.real_to_fat_idx = kwargs.get("real_to_fat_idx", {})

        if self.topology == "fat-tree":
            self.fattree_k = config.get("FATTREE_K")
            self.net_graph = build_fattree(self.fattree_k)

        elif self.topology == "torus3d":
            dims = (
                int(config["TORUS_X"]),
                int(config["TORUS_Y"]),
                int(config["TORUS_Z"])
            )
            wrap = bool(config.get("TORUS_WRAP", True))
            hosts_per_router = int(config.get("HOSTS_PER_ROUTER", config.get("hosts_per_router", 1)))

            # Build the graph and metadata
            self.net_graph, self.meta = build_torus3d(dims, wrap, hosts_per_router=hosts_per_router)

            # Deterministic numeric → host mapping
            X, Y, Z = self.meta["dims"]
            self.id_to_host = {}
            nid = 0
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        for i in range(hosts_per_router):
                            h = f"h_{x}_{y}_{z}_{i}"
                            self.id_to_host[nid] = h
                            nid += 1

        elif self.topology == "dragonfly":
            self.net_graph = build_dragonfly(
                int(config["DRAGONFLY_D"]),
                int(config["DRAGONFLY_A"]),
                int(config.get("DRAGONFLY_P", 1))
            )

        elif self.topology == "capacity":
            # Capacity-only model: no explicit graph
            self.net_graph = None

        else:
            raise ValueError(f"Unsupported topology: {self.topology}")

    def simulate_network_utilization(self, *, job, debug=False):
        net_util = net_cong = net_tx = net_rx = 0
        max_throughput = self.max_link_bw * job.trace_quanta

        if job.nodes_required <= 1:
            # Single node job, skip network impact
            return net_util, net_cong, net_tx, net_rx, max_throughput

        net_tx = get_current_utilization(job.ntx_trace, job)
        net_rx = get_current_utilization(job.nrx_trace, job)
        net_util = network_utilization(net_tx, net_rx, max_throughput)

        if self.topology == "fat-tree":
            host_list = [node_id_to_host_name(n, self.fattree_k) for n in job.scheduled_nodes]
            loads = link_loads_for_job(self.net_graph, host_list, net_tx)
            net_cong = worst_link_util(loads, max_throughput)
            if debug:
                print("  fat-tree hosts:", host_list)

        elif self.topology == "dragonfly":
            D, A, P = self.config["DRAGONFLY_D"], self.config["DRAGONFLY_A"], self.config["DRAGONFLY_P"]
            host_list = [
                dragonfly_node_id_to_host_name(self.real_to_fat_idx[real_n], D, A, P)
                for real_n in job.scheduled_nodes
            ]
            if debug:
                print("  dragonfly hosts:", host_list)
            loads = link_loads_for_job(self.net_graph, host_list, net_tx)
            net_cong = worst_link_util(loads, max_throughput)

        elif self.topology == "torus3d":
            host_list = [self.id_to_host[n] for n in job.scheduled_nodes]
            loads = link_loads_for_job_torus(self.net_graph, self.meta, host_list, net_tx)
            net_cong = worst_link_util(loads, max_throughput)
            if debug:
                print("  torus3d hosts:", host_list)

        elif self.topology == "capacity":
            net_cong = network_congestion(net_tx, net_rx, max_throughput)

        else:
            raise ValueError(f"Unsupported topology: {self.topology}")

        return net_util, net_cong, net_tx, net_rx, max_throughput
