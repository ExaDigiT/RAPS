"""Node placement of scheduled jobs: lowest-numbered free nodes ("first") or scattered ("random")."""
import pytest

from raps.job import Job, job_dict
from raps.policy import PolicyType
from raps.resmgr import make_resource_manager
from raps.resmgr.default import ExclusiveNodeResourceManager

pytestmark = pytest.mark.nodata


def make_job(id, nodes):
    return Job(job_dict(nodes_required=nodes, name=str(id), account="", id=id, cpu_trace=0, gpu_trace=0,
                        ntx_trace=0, nrx_trace=0, submit_time=0, time_limit=3600, start_time=None,
                        end_time=None, expected_run_time=3600))


def runs(nodes):
    """Contiguous runs in a node list."""
    nodes = sorted(nodes)
    return 1 + sum(1 for a, b in zip(nodes, nodes[1:]) if b != a + 1)


def test_first_is_the_default_and_takes_the_lowest_free_nodes():
    rm = ExclusiveNodeResourceManager(100, [3])
    a, b = make_job("a", 5), make_job("b", 5)
    rm.assign_nodes_to_job(a, 0, PolicyType.PRIORITY)
    rm.assign_nodes_to_job(b, 0, PolicyType.PRIORITY)
    assert a.scheduled_nodes == [0, 1, 2, 4, 5] and b.scheduled_nodes == [6, 7, 8, 9, 10]


def test_random_scatters_without_overlap_and_frees_cleanly():
    rm = ExclusiveNodeResourceManager(1000, [0, 1], node_placement="random", seed=5)
    jobs = [make_job(i, 100) for i in range(9)]
    for j in jobs:
        rm.assign_nodes_to_job(j, 0, PolicyType.PRIORITY)
    placed = [n for j in jobs for n in j.scheduled_nodes]
    assert len(placed) == len(set(placed)) == 900 and not {0, 1} & set(placed)   # no node twice, none down
    assert all(runs(j.scheduled_nodes) > 20 for j in jobs)                       # scattered, not blocks
    assert len(rm.available_nodes) == 1000 - 2 - 900
    for j in jobs:
        rm.free_nodes_from_job(j)
    assert rm.available_nodes == list(range(2, 1000))


def test_random_is_reproducible_per_seed():
    def place(seed):
        rm = ExclusiveNodeResourceManager(500, [], node_placement="random", seed=seed)
        j = make_job("x", 50)
        rm.assign_nodes_to_job(j, 0, PolicyType.PRIORITY)
        return j.scheduled_nodes
    assert place(1) == place(1) and place(1) != place(2)


def test_replay_keeps_the_recorded_nodes_whatever_the_placement():
    rm = ExclusiveNodeResourceManager(50, [], node_placement="random", seed=1)
    j = make_job("r", 3)
    j.scheduled_nodes = [10, 11, 40]
    rm.assign_nodes_to_job(j, 0, PolicyType.REPLAY)
    assert j.scheduled_nodes == [10, 11, 40] and 10 not in rm.available_nodes


def test_factory_passes_it_on_and_rejects_nonsense():
    rm = make_resource_manager(10, [], {}, node_placement="random", seed=3)
    assert rm.node_placement == "random"
    with pytest.raises(ValueError):
        ExclusiveNodeResourceManager(10, [], node_placement="shotgun")


def test_twin_placement_changes_nodes_not_start_times():
    from datetime import datetime
    from raps.live.slurm import ClusterState, parse_squeue
    from raps.live.twin import simulate
    from tests.test_live import squeue_line
    now = datetime(2026, 10, 8, 12, 0, 0)
    pn = {"batch": [f"n{i:02d}" for i in range(40)]}
    jobs = parse_squeue("\n".join([
        squeue_line(1, state="PENDING", nodes=10, limit="1:00:00", reason="Priority", priority=9),
        squeue_line(2, state="PENDING", nodes=10, limit="1:00:00", reason="Priority", priority=8),
        squeue_line(3, state="PENDING", nodes=30, limit="1:00:00", reason="Priority", priority=7),
    ]))
    st = ClusterState(user="me", jobs=jobs, node_states={n: "idle" for n in pn["batch"]})
    first, rand = simulate(st, pn, now=now), simulate(st, pn, now=now, placement="random")
    assert first.starts == rand.starts
    names = lambda r, jid: r.runs[jid][2]   # noqa: E731
    assert names(first, "1") == [f"n{i:02d}" for i in range(10)]
    assert names(rand, "1") != names(first, "1") and len(set(names(rand, "1"))) == 10
