"""
Per-node power limits from the RAPS power model (``raps.power.compute_node_power``), used to
scale the console's power colors and to show power as a share of what the machine can draw.
"""
from dataclasses import dataclass

DEFAULT_MAX_W = 2800.0   # until the model has loaded (loading the system config takes ~5 s)


@dataclass(frozen=True)
class NodePowerModel:
    system: str
    max_w: float    # every CPU and GPU at 100% utilization and the NICs busy, with SIVOC losses
    idle_w: float   # everything idle


def load(system):
    from raps.power import compute_node_power
    from raps.system_config import get_system_config
    c = get_system_config(system).get_legacy()
    max_w = compute_node_power(c["CPUS_PER_NODE"], c["GPUS_PER_NODE"], 1, c)[0]
    idle_w = compute_node_power(0, 0, 0, c)[0]
    return NodePowerModel(system, float(max_w), float(idle_w))
