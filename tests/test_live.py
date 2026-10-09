"""Live console: Slurm output parsing, derived numbers, and a pilot run against a fake Slurm."""
import asyncio
from datetime import datetime, timedelta

import pytest

from raps.live import analysis
from raps.live.forecast import Forecaster, HistoryRule, parse_runs
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


def run_line(id, user="me", name="job2", nodes=1, limit_min=120, elapsed=1800, end="2026-10-07T10:00:00",
             energy=None, account="abc123"):
    energy = energy if energy is not None else 1000 * elapsed * nodes  # 1 kW per node
    return SEP.join([str(id), user, account, name, str(nodes), str(limit_min), str(elapsed), end, str(energy)])


RUNS = "\n".join([
    run_line(901, elapsed=1800, end="2026-10-07T09:00:00"),
    run_line(902, elapsed=1800, end="2026-10-07T10:00:00"),
    run_line(903, name="other", elapsed=7200, end="2026-10-07T11:00:00", energy=10 ** 12),  # bad energy
    run_line(904, user="bob", account="xyz", elapsed=60),
    "garbage",
])


def test_history_rule():
    runs = parse_runs(RUNS)
    assert len(runs) == 4 and runs[2].power is None and runs[0].power == 1000
    rule = HistoryRule(runs, days=30)
    jobs = {j.id: j for j in parse_squeue(QUEUE)}
    p = rule.predict(jobs["2"])  # me/job2: two runs at 25% of a 2h limit
    assert p.basis == "job" and p.n == 2 and p.runtime == 1800 and p.power == 1000
    p = rule.predict(jobs["4"])  # me/job4: no runs of that name -> all of my runs
    assert p.basis == "user" and p.n == 3 and p.runtime == 7200 * (0.25 + 0.25 + 1.0) / 3
    p = rule.predict(jobs["1"])  # alice: unknown user, same project as my runs
    assert p.basis == "project"
    assert rule.predict(parse_squeue(squeue_line(9, account="nope"))[0]) is None
    # only the last 5 runs count, and a forecast never exceeds the limit
    many = parse_runs("\n".join(run_line(i, elapsed=60 if i < 5 else 9000, end=f"2026-10-07T{i:02d}:00:00")
                                for i in range(10)))
    assert HistoryRule(many).predict(jobs["2"]).runtime == 7200


def test_forecaster_incremental():
    calls = []

    def fetch(since):
        calls.append(since)
        if len(calls) == 1:
            return parse_runs(RUNS)
        return parse_runs(run_line(905, end=datetime.now().isoformat(timespec="seconds")))
    fc = Forecaster(days=100000, fetch=fetch)
    assert fc.refresh().n_runs == 4
    assert fc.refresh().n_runs == 5 and calls[1] > calls[0]
    fc.days = 1  # runs that ended before the window are dropped
    assert fc.refresh().n_runs == 1


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
        from raps.live.forecast import Forecaster
        fc = Forecaster(days=30, fetch=lambda since: parse_runs(RUNS))
        app = LiveApp(Poller(FakeSlurm(), trend=Trend(None), forecaster=fc))
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
            mj = app.query_one(MyJobs).query_one("#mj-table", DataTable)
            assert mj.row_count == 3
            assert app.poller.latest().forecast is not None
            # job 2 is "me"/"job2" with a 2h limit and history at 25% of the limit -> 30 min, 0 left
            fcst = {mj.get_row_at(i)[0].plain: mj.get_row_at(i)[7].plain for i in range(mj.row_count)}
            assert "past forecast" in fcst["2"] and "job ×2" in fcst["2"]
            assert "user ×3" in fcst["4"]
            await pilot.press("3")
            await pilot.pause(0.2)
            await pilot.press("m")
            await pilot.pause(0.2)
            assert app.query_one(Queue).query_one("#q-table", DataTable).row_count == 3
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert type(app.screen).__name__ == "JobDetailScreen"
    asyncio.run(run())
