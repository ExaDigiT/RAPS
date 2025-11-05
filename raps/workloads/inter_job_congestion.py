import math
import random
from typing import List, Tuple

from raps.job import Job, job_dict
from raps.network import max_throughput_per_tick

class InterJobCongestionWorkload:
    """ Workload generator for inter-job congestion test """
    def inter_job_congestion(self, args) -> List[Job]:
        legacy_cfg = self.config_map[self.partitions[0]]
        topology = legacy_cfg.get("TOPOLOGY", "").lower()
        return generate_jobs(
            legacy_cfg=legacy_cfg,
            topology=topology,
            J=args.numjobs,
            trace_quanta=legacy_cfg.get("TRACE_QUANTA", 20),
            tx_fraction_per_job=getattr(args, 'txfrac', 0.35), # Assuming txfrac might be an arg
            seed=args.seed
        )


def infer_group_params(legacy_cfg: dict, topology: str) -> Tuple[int, int, str]:
    """
    Infer (hosts_per_group, total_groups, group_label)
    depending on network topology.
    """
    total_nodes = int(legacy_cfg["TOTAL_NODES"])

    if topology == "fat-tree":
        k = int(legacy_cfg.get("FATTREE_K", 32))
        H = k // 2  # hosts per ToR
        R = math.ceil(total_nodes / H)
        return H, R, "rack"

    elif topology == "dragonfly":
        routers_per_group = int(legacy_cfg.get("ROUTERS_PER_GROUP", 8))
        nodes_per_router = int(legacy_cfg.get("NODES_PER_ROUTER", 4))
        H = routers_per_group * nodes_per_router
        R = max(1, total_nodes // H)
        return H, R, "group"

    elif topology == "torus3d":
        dims = (
            int(legacy_cfg.get("TORUS_X", 12)),
            int(legacy_cfg.get("TORUS_Y", 12)),
            int(legacy_cfg.get("TORUS_Z", 12)),
        )
        R = math.prod(dims)
        return 1, R, "torus"

    else:
        return 1, 1, "flat"


def pick_two_distinct_groups(R: int) -> Tuple[int, int]:
    """Pick two distinct group indices (far apart if possible)."""
    if R <= 2:
        return (0, 1 if R > 1 else 0)
    a = random.randrange(0, R // 2)
    b = random.randrange(R // 2, R)
    if a == b:
        b = (b + 1) % R
    return a, b


def nodes_in_group(group_idx: int, H: int, total_nodes: int, n: int) -> List[int]:
    """Pick n contiguous nodes from a group."""
    start = group_idx * H
    end = min(start + H, total_nodes)
    n = min(n, end - start)
    base = random.randrange(start, end - n + 1) if (end - start - n) > 0 else start
    return list(range(base, base + n))


def generate_jobs(
    legacy_cfg: dict,
    topology: str,
    J: int = 60,
    trace_quanta: int = 20,
    tx_fraction_per_job: float = 0.35,
    seed: int = 42
) -> List[Job]:
    """Generate synthetic jobs spanning and overlapping local groups."""
    random.seed(seed)
    total_nodes = int(legacy_cfg["TOTAL_NODES"])
    H, R, label = infer_group_params(legacy_cfg, topology)
    per_tick_bw = max_throughput_per_tick(legacy_cfg, trace_quanta)
    per_dir = tx_fraction_per_job * per_tick_bw

    print(f"[INFO] topology={topology}, {label}s={R}, hosts_per_{label}={H}")
    print(f"[INFO] total_nodes={total_nodes}, per-dir={per_dir:.2e} B/tick")

    jobs: List[Job] = []
    jid = 1

    # Roughly 60% cross-group, 25% intra-group, 15% multi-group
    n_cross = int(J * 0.6)
    n_intra = int(J * 0.25)
    n_multi = J - n_cross - n_intra

    for _ in range(n_cross):
        a, b = pick_two_distinct_groups(R)
        nodes = nodes_in_group(a, H, total_nodes, 1) + nodes_in_group(b, H, total_nodes, 1)
        jobs.append(make_job(jid, nodes, per_dir, trace_quanta))
        jid += 1

    for _ in range(n_intra):
        g = random.randrange(0, R)
        nodes = nodes_in_group(g, H, total_nodes, 2)
        jobs.append(make_job(jid, nodes, per_dir, trace_quanta))
        jid += 1

    for _ in range(n_multi):
        a, b = pick_two_distinct_groups(R)
        nodes = nodes_in_group(a, H, total_nodes, 2) + nodes_in_group(b, H, total_nodes, 2)
        jobs.append(make_job(jid, nodes, per_dir, trace_quanta))
        jid += 1

    print(f"[INFO] jobs={len(jobs)} (cross={n_cross}, intra={n_intra}, multi={n_multi})")
    return jobs


def make_job(jid: int, nodes: List[int], per_dir: float, trace_quanta: int) -> Job:
    """Helper: create one synthetic Job object."""
    trace_len = 900 // trace_quanta
    return Job(job_dict(
        id=jid,
        name=f"job_{jid}",
        account="test",
        nodes_required=len(nodes),
        scheduled_nodes=nodes,
        cpu_trace=[0] * trace_len,
        gpu_trace=[0] * trace_len,
        ntx_trace=[per_dir] * trace_len,
        nrx_trace=[per_dir] * trace_len,
        trace_quanta=trace_quanta,
        expected_run_time=900,
        time_limit=1800,
        end_state="COMPLETED"
    ))
