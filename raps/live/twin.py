"""
RAPS start-time forecast for the live queue: replay the current queue forward through RAPS's
scheduler and resource manager, and read off when each pending job starts.

- The machine is split into independent pools of partitions that share nodes (on Frontier: batch
  with batch-spi, extended with extended-spi, g1, service). Each pool gets its own RAPS
  ``ExclusiveNodeResourceManager`` and ``Scheduler`` (priority order with first-fit backfill).
  Jobs bigger than their pool's usable nodes are left out, so the head of a queue can always start.
- Nodes that are down, drained or reserved are left out; running jobs keep their nodes.
- The scheduler plans with time limits, as Slurm does, but jobs actually end at their history-rule
  forecast (``raps.live.forecast``), so nodes free up early the way they do on the real machine.
- Power: running jobs keep their measured power per node (Slurm's CurrentWatts), jobs that
  start get their history-rule power forecast (else the median of running jobs), free nodes draw
  the measured idle mean, and nodes outside the pools or down stay at what they draw now.
- Time advances from one job end to the next (event-driven) rather than RAPS's one-second engine
  ticks, and no power or cooling is modelled, so a full Frontier queue simulates in about a second.

Not modelled: priorities growing with age, jobs submitted from now on, jobs that become eligible
later (dependencies, holds), per-user limits, future reservations, and the batch-spi node subset.
"""
import heapq
import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from raps.job import Job, job_dict
from raps.live.analysis import eligible
from raps.live.slurm import expand_hostlist
from raps.policy import PolicyType
from raps.resmgr.default import ExclusiveNodeResourceManager
from raps.schedulers.default import Scheduler

USABLE = ("idle", "allocated", "mixed", "completing", "planned")
HORIZON = 14 * 86400   # stop simulating after this many seconds


def base_state(state):
    return state.rstrip("*$~#!%@^-+").lower()


@dataclass
class Pool:
    partitions: set
    nodes: list            # node names, sorted


def build_pools(partition_nodes):
    """Group partitions that share any node: {name: [nodes]} -> [Pool]."""
    groups = []
    for name, nodes in partition_nodes.items():
        names, ns = {name}, set(nodes)
        if not ns:
            continue
        for g in [g for g in groups if g[1] & ns]:
            names |= g[0]
            ns |= g[1]
            groups.remove(g)
        groups.append((names, ns))
    return [Pool(names, sorted(ns)) for names, ns in groups]


class FastNodeManager(ExclusiveNodeResourceManager):
    """The RAPS exclusive-node manager with an O(nodes) free (the base class is quadratic)."""

    def free_nodes_from_job(self, job):
        self.available_nodes = sorted(set(self.available_nodes) | set(job.scheduled_nodes))


def raps_job(id, nodes, limit, runtime, priority=0):
    return Job(job_dict(nodes_required=nodes, name=str(id), account="", id=id, priority=priority,
                        cpu_trace=0, gpu_trace=0, ntx_trace=0, nrx_trace=0, submit_time=0,
                        time_limit=limit, start_time=None, end_time=None, expected_run_time=runtime))


@dataclass
class TwinResult:
    at: datetime                                   # when the simulation started ("now")
    starts: dict = field(default_factory=dict)     # pending job id -> datetime, None if not within horizon
    ends: dict = field(default_factory=dict)       # job id -> forecast end (running and pending)
    timeline: list = field(default_factory=list)   # (seconds from now, busy nodes, waiting jobs, node watts)
    power_now: float | None = None                 # measured node watts at the start
    usable: int = 0
    simulated: int = 0                             # pending jobs put through the scheduler
    seconds: float = 0.0                           # wall time the simulation took

    def _at(self, t, k):
        """Column k of the timeline t seconds from now (a step function)."""
        v = self.timeline[0][k] if self.timeline else None
        for row in self.timeline:
            if row[0] > t:
                break
            v = row[k]
        return v

    def busy_at(self, t):
        return self._at(t, 1) or 0

    def waiting_at(self, t):
        return self._at(t, 2) or 0

    def power_at(self, t):
        """Forecast node watts t seconds from now, None without power readings."""
        return self._at(t, 3)


def _runtime(state, job):
    """Forecast run time (history rule), else the time limit."""
    p = state.predict(job)
    return p.runtime if p is not None else job.limit


def simulate(state, partition_nodes, horizon=HORIZON, now=None):
    """Run the live queue in ``state`` forward through RAPS. Returns a ``TwinResult``."""
    t_wall = time.perf_counter()
    now = now or datetime.now()
    res = TwinResult(at=now)
    pools = build_pools(partition_nodes)
    pool_of_part = {p: i for i, pool in enumerate(pools) for p in pool.partitions}
    pool_of_node = {n: i for i, pool in enumerate(pools) for n in pool.nodes}
    node_states = state.node_states or {}
    watts = state.node_power or {}
    has_power = bool(watts)

    running = [[] for _ in pools]
    queued = [[] for _ in pools]
    for j in state.jobs or []:
        if j.running and j.nodelist and j.limit:
            nodes = expand_hostlist(j.nodelist)
            i = pool_of_node.get(nodes[0]) if nodes else None
            if i is not None:
                running[i].append((j, nodes))
        elif eligible(j) and j.limit:
            i = next((pool_of_part[p] for p in j.partition.split(",") if p in pool_of_part), None)
            if i is not None:
                queued[i].append(j)

    # power per node: measured for running jobs, forecast (else the running median) for new ones
    job_w = {}
    for i in range(len(pools)):
        for j, nodes in running[i]:
            w = [watts[n] for n in nodes if watts.get(n)]
            if w:
                job_w[j.id] = sum(w) / len(w)
    typical_w = sorted(job_w.values())[len(job_w) // 2] if job_w else 1100.0
    cap_w = state.node_max_w  # a forecast never exceeds what a node can draw
    for i in range(len(pools)):
        for j, _ in running[i]:
            if j.id not in job_w:
                p = state.predict(j)
                job_w[j.id] = min(p.power, cap_w) if p is not None and p.power else typical_w
        for j in queued[i]:
            p = state.predict(j)
            job_w[j.id] = min(p.power, cap_w) if p is not None and p.power else typical_w
    idle = [watts[n] for n, s in node_states.items() if watts.get(n) and base_state(s) in ("idle", "planned")]
    idle_w = sum(idle) / len(idle) if idle else 480.0
    fixed_w = sum(w for n, w in watts.items() if n not in pool_of_node)   # outside every pool
    fixed_w += (len(node_states) - len(watts)) * idle_w if has_power else 0  # no valid reading
    run_w = [0.0]                                                        # running jobs' total

    events = []  # (end time, seq, pool, raps job)
    seq = itertools.count()
    pending_ids = {}
    pools_state = []
    for i, pool in enumerate(pools):
        busy_names = {n for _, nodes in running[i] for n in nodes}
        down = [k for k, n in enumerate(pool.nodes)
                if n not in busy_names and base_state(node_states.get(n, "down")) not in USABLE]
        fixed_w += sum(watts.get(pool.nodes[k], 0) for k in down)
        rm = FastNodeManager(len(pool.nodes), down)
        sched = Scheduler(config={}, policy="priority", bfpolicy="firstfit", resource_manager=rm)
        res.usable += len(rm.available_nodes)
        run_list = []
        for j, _ in running[i]:
            start = (j.start - now).total_seconds() if j.start else -j.elapsed
            p = state.predict(j)
            # a job that has outrun its forecast is assumed to run to its limit
            runtime = p.runtime if p is not None and p.runtime > j.elapsed else j.limit
            rj = raps_job(j.id, j.nodes, j.limit, max(runtime, j.elapsed + 1))
            try:
                rm.assign_nodes_to_job(rj, start, PolicyType.PRIORITY)
            except ValueError:  # squeue and sinfo disagree for a moment; skip the job
                continue
            run_list.append(rj)
            run_w[0] += rj.nodes_required * job_w[j.id]
            res.ends[j.id] = now + timedelta(seconds=rj.end_time)
            heapq.heappush(events, (rj.end_time, next(seq), i, rj))
        queue = []
        for j in queued[i]:
            if j.nodes > len(pool.nodes) - len(down):
                res.starts[j.id] = None  # bigger than the usable pool
                continue
            queue.append(raps_job(j.id, j.nodes, j.limit, _runtime(state, j), priority=j.priority))
            pending_ids[j.id] = i
        res.simulated += len(queue)
        pools_state.append((rm, sched, run_list, queue))

    def schedule_all(t):
        for i, (rm, sched, run_list, queue) in enumerate(pools_state):
            if not queue:
                continue
            before = len(run_list)
            sched.schedule(queue, run_list, t)
            for rj in run_list[before:]:
                run_w[0] += rj.nodes_required * job_w[rj.id]
                res.starts[rj.id] = now + timedelta(seconds=t)
                res.ends[rj.id] = now + timedelta(seconds=rj.end_time)
                heapq.heappush(events, (rj.end_time, next(seq), i, rj))

    def sample(t):
        busy = sum(len(rm.nodes) - len(rm.down_nodes) - len(rm.available_nodes) for rm, *_ in pools_state)
        free = sum(len(rm.available_nodes) for rm, *_ in pools_state)
        power = fixed_w + run_w[0] + free * idle_w if has_power else None
        res.timeline.append((t, busy, sum(len(q) for *_, q in pools_state), power))

    t = 0.0
    schedule_all(t)
    sample(t)
    # keep going after the queue is empty so the timeline shows running jobs ending (with no new
    # submissions, which the twin does not model, the machine then drains)
    while events:
        t = events[0][0]
        if t > horizon:
            break
        while events and events[0][0] <= t:
            _, _, i, rj = heapq.heappop(events)
            rm, _, run_list, _ = pools_state[i]
            rm.free_nodes_from_job(rj)
            run_list.remove(rj)
            run_w[0] -= rj.nodes_required * job_w[rj.id]
        schedule_all(t)
        sample(t)
    res.power_now = state.power
    for jid in pending_ids:
        res.starts.setdefault(jid, None)
    res.seconds = time.perf_counter() - t_wall
    return res


class Twin:
    """What the poller calls: fetches the partition layout once, then simulates on each refresh."""

    def __init__(self, slurm, get_state, horizon=HORIZON):
        self.slurm = slurm
        self.get_state = get_state
        self.horizon = horizon
        self.partition_nodes = None

    def refresh(self):
        state = self.get_state()
        if state.jobs is None or state.node_states is None:
            return None
        if self.partition_nodes is None:
            self.partition_nodes = self.slurm.partition_nodes()
        return simulate(state, self.partition_nodes, self.horizon)
