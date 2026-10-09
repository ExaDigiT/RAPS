"""
Per-job runtime and power forecasts for the live console: the history rule from ``raps.forecast``.

A job's runtime is forecast as its time limit times the mean runtime / limit ratio of the last 5
finished runs with the same user and job name, falling back to the same user, then the same
account. Power per node is the mean of the same runs' energy / runtime / nodes. On two-plus years
of Frontier jobs this halves Slurm's typical error (see ``docs/forecast-results.md``).

History comes from ``sacct -a`` over the last ``days`` days (about 13 s for 30 days on Frontier),
then incrementally. It is held in memory only and never written to disk, since it holds every
user's job names.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from raps.live.slurm import SEP, parse_slurm_time, run

LAST_N = 5
RATIO_MAX = 1.2                      # same clip as raps.forecast.dataset
POWER_MIN_W, POWER_MAX_W = 50, 5000  # plausible mean watts per Frontier node
FINISHED_STATES = "CD,TO,F,CA,OOM,NF"
FIELDS = "JobIDRaw,User,Account,JobName,NNodes,TimelimitRaw,ElapsedRaw,End,ConsumedEnergyRaw"
BASES = ("job", "user", "project")   # most to least specific


@dataclass
class Run:
    """One finished job, reduced to what the history rule needs."""
    id: str
    user: str
    account: str
    name: str
    end: datetime
    ratio: float          # runtime / time limit, clipped to [0, RATIO_MAX]
    power: float | None   # mean watts per node, None if missing or implausible


@dataclass
class Prediction:
    runtime: float        # forecast seconds from start to end (at most the time limit)
    power: float | None   # forecast mean watts per node
    basis: str            # "job" (same user + name), "user" or "project"
    n: int                # earlier runs in that group (up to LAST_N used)


def parse_runs(text):
    runs = []
    for line in text.splitlines():
        f = line.split(SEP)
        if len(f) != 9:
            continue
        jid, user, account, name, nodes, limit_min, elapsed, end, energy = f
        try:
            nodes, limit, elapsed = int(nodes), int(limit_min) * 60, int(elapsed)
        except ValueError:
            continue
        end = parse_slurm_time(end)
        if end is None or nodes <= 0 or limit <= 0 or elapsed <= 0:
            continue
        try:
            power = float(energy) / elapsed / nodes
        except ValueError:
            power = None
        if power is not None and not POWER_MIN_W <= power <= POWER_MAX_W:
            power = None
        runs.append(Run(id=jid, user=user, account=account, name=name, end=end,
                        ratio=min(elapsed / limit, RATIO_MAX), power=power))
    return runs


def fetch_runs(since, until=None):
    """Every user's jobs that finished in [since, until] (``sacct -a``)."""
    # with -s, sacct needs an explicit end time (otherwise the window is empty)
    until = until or datetime.now()
    cmd = ["sacct", "-a", "-X", "-n", "-P", f"--delimiter={SEP}", "-s", FINISHED_STATES,
           "-S", since.strftime("%Y-%m-%dT%H:%M:%S"), "-E", until.strftime("%Y-%m-%dT%H:%M:%S"), "-o", FIELDS]
    return parse_runs(run(cmd, timeout=180))


class HistoryRule:
    """An immutable index of the last LAST_N runs per (user, name), user and account."""

    def __init__(self, runs, days=None, updated=None):
        self.days = days
        self.updated = updated
        self.n_runs = 0
        groups = {b: defaultdict(lambda: deque(maxlen=LAST_N)) for b in BASES}
        for r in sorted(runs, key=lambda r: r.end):
            self.n_runs += 1
            for b, key in zip(BASES, ((r.user, r.name), r.user, r.account)):
                groups[b][key].append(r)
        self.count = {b: defaultdict(int) for b in BASES}
        for r in runs:
            self.count["job"][(r.user, r.name)] += 1
            self.count["user"][r.user] += 1
            self.count["project"][r.account] += 1
        self.groups = {b: dict(g) for b, g in groups.items()}

    def predict(self, job):
        """A ``Prediction`` for a squeue ``Job``, or None without a time limit or any history."""
        if not job.limit:
            return None
        for b, key in zip(BASES, ((job.user, job.name), job.user, job.account)):
            last = self.groups[b].get(key)
            if last:
                ratio = sum(r.ratio for r in last) / len(last)
                powers = [r.power for r in last if r.power is not None]
                return Prediction(runtime=min(ratio, 1.0) * job.limit,
                                  power=sum(powers) / len(powers) if powers else None,
                                  basis=b, n=self.count[b][key])
        return None


def remaining(job, pred):
    """Forecast seconds left for a running job; None once it has outrun its forecast."""
    left = pred.runtime - job.elapsed
    return left if left > 0 else None


class Forecaster:
    """Keeps ``days`` of finished runs and publishes a fresh ``HistoryRule`` on each refresh."""

    def __init__(self, days=30, fetch=fetch_runs):
        self.days = days
        self.fetch = fetch
        self.runs = {}
        self.last = None

    def refresh(self):
        now = datetime.now()
        # after the first full pull, re-read the last hour so nothing is missed between polls
        since = now - timedelta(days=self.days) if self.last is None else self.last - timedelta(hours=1)
        for r in self.fetch(since):
            self.runs[r.id] = r
        cutoff = now - timedelta(days=self.days)
        self.runs = {k: r for k, r in self.runs.items() if r.end >= cutoff}
        self.last = now
        return HistoryRule(self.runs.values(), days=self.days, updated=now)
