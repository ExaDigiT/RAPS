"""Backfill shadow time in the default and experimental schedulers."""
import pytest

from raps.job import Job, job_dict
from raps.policy import PolicyType
from raps.resmgr.default import ExclusiveNodeResourceManager
from raps.schedulers import default, experimental

pytestmark = pytest.mark.nodata


def make_job(id, nodes, limit, runtime=None):
    return Job(job_dict(nodes_required=nodes, name=str(id), account="", id=id, cpu_trace=0, gpu_trace=0,
                        ntx_trace=0, nrx_trace=0, submit_time=0, time_limit=limit, start_time=None,
                        end_time=None, expected_run_time=runtime or limit))


@pytest.mark.parametrize("module", [default, experimental])
def test_shadow_time_uses_when_running_jobs_reach_their_limits(module):
    rm = ExclusiveNodeResourceManager(10, [])
    sched = module.Scheduler(config={}, policy="priority", bfpolicy="firstfit", resource_manager=rm)
    # A: 4 nodes, 10 h limit, started 9 h ago -> limit reached at t=+1 h
    # B: 4 nodes, 2 h limit, just started     -> limit reached at t=+2 h
    a, b = make_job("A", 4, 10 * 3600), make_job("B", 4, 2 * 3600)
    rm.assign_nodes_to_job(a, -9 * 3600, PolicyType.PRIORITY)
    rm.assign_nodes_to_job(b, 0, PolicyType.PRIORITY)
    running = [a, b]
    # The head job needs 6 nodes: 2 free now + A's 4 at t=+1 h, so nothing may backfill past +1 h
    head = make_job("head", 6, 3600)
    short = make_job("short", 2, 1800)   # ends at +30 min: may backfill
    late = make_job("late", 2, 5400)     # ends at +90 min: would delay the head job
    assert sched.find_backfill_job([head, late, short], running, 0) is short
    assert sched.find_backfill_job([head, late], running, 0) is None


def setup(module, bfpolicy, total, running_specs):
    """running_specs: (id, nodes, limit, started at) -> (scheduler, running jobs)."""
    rm = ExclusiveNodeResourceManager(total, [])
    sched = module.Scheduler(config={}, policy="priority", bfpolicy=bfpolicy, resource_manager=rm)
    running = []
    for id, nodes, limit, start in running_specs:
        j = make_job(id, nodes, limit)
        rm.assign_nodes_to_job(j, start, PolicyType.PRIORITY)
        running.append(j)
    return sched, running


@pytest.mark.parametrize("module", [default, experimental])
@pytest.mark.parametrize("bfpolicy", ["firstfit", "easy"])
def test_extra_nodes_may_run_past_the_shadow_time(module, bfpolicy):
    # 12 nodes: A holds 8 until +1 h, 4 are free. The head job needs 10, so at +1 h it gets
    # 12 nodes, 2 more than it needs: a long job on at most 2 nodes cannot delay it.
    sched, running = setup(module, bfpolicy, 12, [("A", 8, 3600, 0)])
    head, wide, narrow, narrow2 = (make_job("head", 10, 3600), make_job("wide", 3, 5 * 3600),
                                   make_job("narrow", 2, 5 * 3600), make_job("narrow2", 2, 5 * 3600))
    queue = [head, wide, narrow, narrow2]
    assert sched.find_backfill_job(queue, running, 0) is narrow       # wide would need 3 of the 2 extra
    sched.backfill(queue, running, 0)
    assert [j.id for j in queue] == ["head", "wide", "narrow2"]       # narrow2 would use nodes head needs
    assert [j.id for j in running] == ["A", "narrow"]


@pytest.mark.parametrize("module", [default, experimental])
def test_easy_does_not_reorder_the_queue(module):
    # EASY tries candidates in submit order but must keep protecting the priority head job
    sched, running = setup(module, "easy", 10, [("A", 6, 3600, 0)])
    head = make_job("head", 10, 3600)
    head.submit_time = 500                                   # submitted last, highest priority
    early_long = make_job("early_long", 2, 5 * 3600)         # submitted first: would delay head
    early_long.submit_time = 0
    short = make_job("short", 2, 1800)
    short.submit_time = 100
    queue = [head, early_long, short]
    sched.backfill(queue, running, 0)
    assert [j.id for j in queue] == ["head", "early_long"]   # queue order kept; head still first
    assert [j.id for j in running] == ["A", "short"]


@pytest.mark.parametrize("module", [default, experimental])
def test_head_job_that_can_never_fit_does_not_block_backfill(module):
    sched, running = setup(module, "firstfit", 4, [("A", 2, 3600, 0)])
    queue = [make_job("huge", 50, 3600), make_job("long", 2, 10 * 3600)]
    assert sched.find_backfill_job(queue, running, 0).id == "long"
