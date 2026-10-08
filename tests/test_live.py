"""Live console: Slurm output parsing, derived numbers, and a pilot run against a fake Slurm."""
import asyncio
from datetime import datetime, timedelta

import pytest

from raps.live import analysis
from raps.live.slurm import (
    SEP, ClusterState, Slurm, expand_hostlist, parse_sacct, parse_showusage, parse_sinfo_nodes,
    parse_sinfo_partitions, parse_slurm_duration, parse_squeue,
)

pytestmark = pytest.mark.nodata


def squeue_line(id, user="alice", state="RUNNING", nodes=2, limit="2:00:00", elapsed="30:00", start="N/A",
                reason="None", priority=100, nodelist="", account="abc123", partition="batch"):
    return SEP.join([str(id), user, account, f"job{id}", state, str(nodes), limit, elapsed, "2026-10-08T10:00:00",
                     start, reason, partition, "normal", str(priority), nodelist])


QUEUE = "\n".join([
    squeue_line(1, nodes=2, nodelist="frontier[00001-00002]"),
    squeue_line(2, user="me", nodes=1, nodelist="frontier00130"),
    squeue_line(3, user="bob", state="PENDING", nodes=8, reason="Priority", priority=300, elapsed="0:00"),
    squeue_line(4, user="me", state="PENDING", nodes=4, reason="Priority", priority=200, elapsed="0:00",
                start=(datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")),
    squeue_line(5, user="me", state="PENDING", nodes=4, reason="Dependency", priority=999, elapsed="0:00"),
])
NODES = "\n".join(["frontier00001|allocated", "frontier00002|allocated", "frontier00003|idle",
                   "frontier00130|allocated", "frontier00131|drained*", "frontier00132|planned"])


def test_durations():
    assert parse_slurm_duration("1-02:03:04") == 93784
    assert parse_slurm_duration("02:03:04") == 7384
    assert parse_slurm_duration("03:04") == 184
    assert parse_slurm_duration("UNLIMITED") is None


def test_hostlist():
    assert expand_hostlist("frontier[00008-00010,00020],login1") == [
        "frontier00008", "frontier00009", "frontier00010", "frontier00020", "login1"]
    assert expand_hostlist("") == []


def test_parse_squeue_and_summary():
    jobs = parse_squeue(QUEUE)
    assert [j.id for j in jobs] == ["1", "2", "3", "4", "5"]
    assert jobs[0].limit == 7200 and jobs[0].elapsed == 1800
    state = ClusterState(user="me", jobs=jobs, node_states=parse_sinfo_nodes(NODES))
    s = analysis.summarize(state)
    assert (s.total_nodes, s.busy_nodes, s.idle_nodes, s.out_nodes, s.planned_nodes) == (6, 3, 1, 1, 1)
    assert (s.run_jobs, s.pend_jobs, s.eligible_jobs, s.blocked_jobs) == (2, 3, 2, 1)
    assert (s.my_run, s.my_pend, s.my_nodes) == (1, 2, 1)
    assert s.backlog_nh == pytest.approx(8 * 2 + 4 * 2)


def test_queue_position():
    jobs = parse_squeue(QUEUE)
    qp = analysis.queue_position(jobs[3], jobs)
    # only job 3 (eligible, higher priority) is ahead; job 5 is blocked on a dependency
    assert (qp.rank, qp.of, qp.ahead_nodes) == (2, 2, 8)
    assert analysis.queue_position(jobs[0], jobs) is None


def test_node_job_map():
    m = analysis.node_job_map(parse_squeue(QUEUE))
    assert m["frontier00001"].id == "1" and m["frontier00130"].id == "2"


def test_showusage_and_partitions_and_sacct():
    usage = parse_showusage("""
frontier usage for the project's current allocation period:
                                  Project Totals          me
 Project      Allocation        Usage    Remaining          Usage
__________________________|____________________________|_____________
 abc123           20000   |      37879       -17879  |          5
 xyz999-aimo              |          0                |          0
""")
    assert usage[0].project == "abc123" and usage[0].allocation == 20000 and usage[0].my_usage == 5
    assert usage[1].allocation is None
    parts = parse_sinfo_partitions("batch*|up|9536|8927/569/40/9536|7-00:00:00")
    assert parts[0].name == "batch" and parts[0].default and parts[0].idle == 569
    hist = parse_sacct("5621275|prof|gen053|batch|CANCELLED by 1|1|320|45|2026-10-05T11:35:14|2026-10-05T11:40:34|0:0")
    assert hist[0].state == "CANCELLED" and hist[0].limit == 2700


def test_plan_cabinets_fits():
    from raps.live.widgets import plan_cabinets
    for w, h in [(200, 90), (100, 40), (40, 10)]:
        tw, th, k, cols, scale = plan_cabinets(77, w, h)
        rows = -(-77 // cols)
        assert cols * (tw * scale + 1) - 1 <= w and rows * (th * scale + 1) - 1 <= h
        assert tw * th * k == 128


class FakeSlurm(Slurm):
    def __init__(self):
        super().__init__(user="me")

    def squeue(self):
        return parse_squeue(QUEUE)

    def node_states(self):
        return parse_sinfo_nodes(NODES)

    def partitions(self):
        return parse_sinfo_partitions("batch*|up|6|3/1/2/6|7-00:00:00")

    def allocations(self):
        return []

    def fairshare(self):
        return []

    def history(self):
        return []

    def reservations(self):
        return []

    def job_detail(self, job_id):
        return f"JobId={job_id} JobName=job{job_id}\n   UserId=me(1) JobState=RUNNING Reason=None"

    @staticmethod
    def cluster_name():
        return "fake"


def test_app_pilot():
    pytest.importorskip("textual")
    from raps.live.app import LiveApp
    from raps.live.poller import Poller, Trend
    from raps.live.views import MyJobs, Queue
    from textual.widgets import DataTable

    async def run():
        app = LiveApp(Poller(FakeSlurm(), trend=Trend(None)))
        async with app.run_test(size=(160, 48)) as pilot:
            for _ in range(40):
                await pilot.pause(0.1)
                if app.poller.latest().reservations is not None:
                    break
            await pilot.pause(0.3)
            for key in "123456":
                await pilot.press(key)
                await pilot.pause(0.1)
            # the shell prompt takes focus: digits type into it, F-keys still switch tabs
            await pilot.press("2")
            assert app.query_one("TabbedContent").active == "shell"
            await pilot.press("f2")
            await pilot.pause(0.2)
            assert app.query_one(MyJobs).query_one("#mj-table", DataTable).row_count == 3
            await pilot.press("3")
            await pilot.pause(0.2)
            await pilot.press("m")
            await pilot.pause(0.2)
            assert app.query_one(Queue).query_one("#q-table", DataTable).row_count == 3
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert type(app.screen).__name__ == "JobDetailScreen"
    asyncio.run(run())
