"""
Derived numbers for the live console, computed from a ``ClusterState``.

Kept free of UI code: this is the layer a RAPS forecast will plug into (e.g. replace Slurm's
backfill estimate in ``queue_position`` with a simulated start time).
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from raps.live.slurm import expand_hostlist

# Pending reasons that mean "waiting on the scheduler" rather than blocked on something else
ELIGIBLE_REASONS = {"Priority", "Resources", "None", "ReqNodeNotAvail"}
SIZE_BINS = [(1, 1), (2, 15), (16, 127), (128, 1023), (1024, 4095), (4096, 10 ** 9)]


def size_label(lo, hi):
    if lo == hi:
        return str(lo)
    if hi >= 10 ** 9:
        return f"{lo}+"
    return f"{lo}–{hi}"


def eligible(job):
    return job.pending and (job.reason in ELIGIBLE_REASONS or job.reason.startswith("ReqNodeNotAvail"))


def node_job_map(jobs):
    """node name -> running Job."""
    out = {}
    for j in jobs or []:
        if j.running and j.nodelist:
            for n in expand_hostlist(j.nodelist):
                out[n] = j
    return out


@dataclass
class Summary:
    total_nodes: int = 0
    busy_nodes: int = 0
    idle_nodes: int = 0
    out_nodes: int = 0          # down, drained, maint, reboot
    planned_nodes: int = 0      # held idle for a large pending job
    run_jobs: int = 0
    run_nodes: int = 0
    pend_jobs: int = 0
    eligible_jobs: int = 0
    blocked_jobs: int = 0
    backlog_nh: float = 0.0     # requested node-hours of eligible pending jobs
    my_run: int = 0
    my_pend: int = 0
    my_nodes: int = 0

    @property
    def usable_nodes(self):
        return max(self.total_nodes - self.out_nodes, 0)

    @property
    def utilization(self):
        return self.busy_nodes / self.usable_nodes if self.usable_nodes else 0.0

    @property
    def drain_hours(self):
        """Hours the eligible backlog would take with every usable node working on it."""
        return self.backlog_nh / self.usable_nodes if self.usable_nodes else 0.0


OUT_STATES = ("down", "drain", "drained", "draining", "maint", "reboot", "fail", "failing")


def summarize(state):
    from raps.live.widgets import base_state  # light import; avoids a cycle at module load
    s = Summary()
    if state.node_states:
        for st in state.node_states.values():
            b = base_state(st)
            s.total_nodes += 1
            if b in ("allocated", "mixed", "completing"):
                s.busy_nodes += 1
            elif b == "idle":
                s.idle_nodes += 1
            elif b == "planned":
                s.planned_nodes += 1
            elif b in OUT_STATES:
                s.out_nodes += 1
    for j in state.jobs or []:
        mine = j.user == state.user
        if j.running:
            s.run_jobs += 1
            s.run_nodes += j.nodes
            if mine:
                s.my_run += 1
                s.my_nodes += j.nodes
        elif j.pending:
            s.pend_jobs += 1
            if mine:
                s.my_pend += 1
            if eligible(j):
                s.eligible_jobs += 1
                s.backlog_nh += j.node_hours
            else:
                s.blocked_jobs += 1
    if not state.node_states:  # before sinfo -N arrives, estimate from the queue alone
        s.busy_nodes = s.run_nodes
    return s


@dataclass
class QueuePosition:
    rank: int          # 1 = next in line among eligible jobs in the partition
    of: int
    ahead_nh: float    # node-hours requested by the jobs ahead of it
    ahead_nodes: int


def queue_position(job, jobs):
    """Where a pending job sits in its partition's priority order (eligible jobs only)."""
    if not job.pending:
        return None
    peers = [j for j in jobs if eligible(j) and j.partition == job.partition]
    ahead = [j for j in peers if j.priority > job.priority and j.id != job.id]
    return QueuePosition(rank=len(ahead) + 1, of=len(peers) + (0 if eligible(job) else 1),
                         ahead_nh=sum(j.node_hours for j in ahead), ahead_nodes=sum(j.nodes for j in ahead))


def size_histogram(jobs):
    """[(label, running jobs, running nodes, pending jobs, pending nodes)] by job size."""
    rows = []
    for lo, hi in SIZE_BINS:
        r = [j for j in jobs if j.running and lo <= j.nodes <= hi]
        p = [j for j in jobs if eligible(j) and lo <= j.nodes <= hi]
        rows.append((size_label(lo, hi), len(r), sum(j.nodes for j in r), len(p), sum(j.nodes for j in p)))
    return rows


def top_accounts(jobs, n=8, key="account"):
    """[(name, running nodes, running jobs, pending node-hours)] sorted by running nodes."""
    run_nodes, run_jobs, pend_nh = Counter(), Counter(), defaultdict(float)
    for j in jobs:
        k = getattr(j, key)
        if j.running:
            run_nodes[k] += j.nodes
            run_jobs[k] += 1
        elif eligible(j):
            pend_nh[k] += j.node_hours
    names = sorted(set(run_nodes) | set(pend_nh), key=lambda a: (-run_nodes[a], -pend_nh[a]))
    return [(a, run_nodes[a], run_jobs[a], pend_nh[a]) for a in names[:n]]


def reason_counts(jobs):
    return Counter(j.reason for j in jobs if j.pending).most_common()


@dataclass
class ForecastSummary:
    covered: int = 0          # eligible pending jobs with a forecast
    eligible: int = 0
    backlog_nh: float = 0.0   # eligible backlog with forecast runtimes (limit where none)
    limit_nh: float = 0.0     # the same backlog at requested limits
    free_1h_fcst: int = 0     # nodes running jobs are forecast to release within the hour
    free_1h_limit: int = 0    # ... if every job ran to its limit

    @property
    def coverage(self):
        return self.covered / self.eligible if self.eligible else 0.0


def forecast_summary(state, horizon=3600):
    """How the history-rule forecast changes the backlog and the nodes freeing up soon."""
    s = ForecastSummary()
    for j in state.jobs or []:
        p = state.predict(j)
        if eligible(j):
            s.eligible += 1
            s.limit_nh += j.node_hours
            if p is not None:
                s.covered += 1
                s.backlog_nh += j.nodes * p.runtime / 3600
            else:
                s.backlog_nh += j.node_hours
        elif j.running and j.limit:
            if j.remaining <= horizon:
                s.free_1h_limit += j.nodes
            # a job that has outrun its forecast is assumed to run to its limit
            left = p.runtime - j.elapsed if p is not None else -1
            if (left if left > 0 else j.remaining) <= horizon:
                s.free_1h_fcst += j.nodes
    return s
