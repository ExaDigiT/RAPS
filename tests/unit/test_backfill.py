"""Backfill shadow time in the default scheduler."""
import pytest

from raps.job import Job, job_dict
from raps.policy import PolicyType
from raps.resmgr.default import ExclusiveNodeResourceManager
from raps.schedulers.default import Scheduler

pytestmark = pytest.mark.nodata


def make_job(id, nodes, limit, runtime=None):
    return Job(job_dict(nodes_required=nodes, name=str(id), account="", id=id, cpu_trace=0, gpu_trace=0,
                        ntx_trace=0, nrx_trace=0, submit_time=0, time_limit=limit, start_time=None,
                        end_time=None, expected_run_time=runtime or limit))


def test_shadow_time_uses_when_running_jobs_reach_their_limits():
    rm = ExclusiveNodeResourceManager(10, [])
    sched = Scheduler(config={}, policy="priority", bfpolicy="firstfit", resource_manager=rm)
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
