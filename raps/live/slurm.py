"""
Slurm data collection for the live console: run the CLI tools and parse their output.

Everything here is plain Python with no Textual dependency. The ``parse_*`` functions take the raw
text a command printed, so they can be tested with captured output and reused by RAPS (e.g. to seed
a forecast from the live queue).
"""
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

SEP = "\x1f"  # unit separator: never appears in job names, unlike "|"

# squeue fields: (header, format spec)
SQUEUE_FIELDS = [
    ("id", "%i"), ("user", "%u"), ("account", "%a"), ("name", "%j"), ("state", "%T"),
    ("nodes", "%D"), ("limit", "%l"), ("elapsed", "%M"), ("submit", "%V"), ("start", "%S"),
    ("reason", "%r"), ("partition", "%P"), ("qos", "%q"), ("priority", "%Q"), ("nodelist", "%N"),
]

# ---------------------------------------------------------------------------------------------
# small parsers
# ---------------------------------------------------------------------------------------------


def parse_slurm_duration(text):
    """'1-02:03:04', '02:03:04', '03:04', '45' (minutes) -> seconds. None for UNLIMITED/N/A."""
    text = (text or "").strip()
    if not text or text in ("UNLIMITED", "N/A", "INVALID", "NOT_SET", "Partition_Limit"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(float(p)) for p in text.split(":")]
    if len(parts) == 1:  # bare number in a time field is minutes
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def parse_slurm_time(text):
    """ISO timestamp from Slurm, or None for N/A/Unknown/None."""
    text = (text or "").strip()
    if not text or text in ("N/A", "Unknown", "None", "(null)"):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def expand_hostlist(expr):
    """
    'frontier[00001-00003,00007],login1' -> ['frontier00001', 'frontier00002', 'frontier00003',
    'frontier00007', 'login1']. Zero padding in a range is preserved.
    """
    out = []
    if not expr or expr in ("(null)", "None"):
        return out
    # split on commas that are not inside brackets
    depth, cur, items = 0, [], []
    for ch in expr:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    for item in items:
        m = re.fullmatch(r"([^\[]*)\[([^\]]+)\](.*)", item)
        if not m:
            if item:
                out.append(item)
            continue
        prefix, body, suffix = m.groups()
        for part in body.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                width = len(lo)
                out.extend(f"{prefix}{i:0{width}d}{suffix}" for i in range(int(lo), int(hi) + 1))
            else:
                out.append(f"{prefix}{part}{suffix}")
    return out


def node_index(name):
    """Trailing integer of a node name ('frontier04365' -> 4365), or None."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------
# data records
# ---------------------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    user: str
    account: str
    name: str
    state: str
    nodes: int
    limit: int | None        # seconds
    elapsed: int             # seconds
    submit: datetime | None
    start: datetime | None   # actual start, or Slurm's backfill estimate for pending jobs
    reason: str
    partition: str
    qos: str
    priority: int
    nodelist: str

    @property
    def pending(self):
        return self.state == "PENDING"

    @property
    def running(self):
        return self.state in ("RUNNING", "COMPLETING")

    @property
    def remaining(self):
        return None if self.limit is None else max(self.limit - self.elapsed, 0)

    @property
    def node_hours(self):
        """Requested node-hours (limit x nodes); what the job will cost if it runs to its limit."""
        return self.nodes * (self.limit or 0) / 3600.0


@dataclass
class Partition:
    name: str
    default: bool
    avail: str
    total: int
    alloc: int
    idle: int
    other: int
    timelimit: str


@dataclass
class Allocation:
    project: str
    allocation: float | None   # node-hours
    usage: float
    remaining: float | None
    my_usage: float


@dataclass
class FairShare:
    account: str
    raw_usage: int
    fairshare: float


@dataclass
class HistJob:
    id: str
    name: str
    account: str
    partition: str
    state: str
    nodes: int
    elapsed: int
    limit: int | None
    start: datetime | None
    end: datetime | None
    exit_code: str


@dataclass
class Reservation:
    name: str
    start: datetime | None
    end: datetime | None
    nodes: int
    state: str
    flags: str
    users: str


@dataclass
class ClusterState:
    """Everything the console knows at one instant. Fields stay None until first fetched."""
    user: str
    cluster: str = ""
    jobs: list[Job] | None = None
    node_states: dict[str, str] | None = None
    partitions: list[Partition] | None = None
    allocations: list[Allocation] | None = None
    fairshare: list[FairShare] | None = None
    history: list[HistJob] | None = None
    reservations: list[Reservation] | None = None
    node_power: dict[str, int] | None = None  # node -> current watts
    forecast: object | None = None    # raps.live.forecast.HistoryRule
    twin: object | None = None        # raps.live.twin.TwinResult
    updated: dict[str, datetime] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def predict(self, job):
        return self.forecast.predict(job) if self.forecast is not None else None

    @property
    def power(self):
        """Measured node watts, None until scontrol show node arrives."""
        return power_total(self.node_power, len(self.node_states or {}))

    @property
    def my_jobs(self):
        return [j for j in (self.jobs or []) if j.user == self.user]


# ---------------------------------------------------------------------------------------------
# parsers for whole command outputs
# ---------------------------------------------------------------------------------------------


def _int(text, default=0):
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def parse_squeue(text):
    jobs, seen = [], set()
    n = len(SQUEUE_FIELDS)
    for line in text.splitlines():
        parts = line.split(SEP)
        if len(parts) != n:
            continue
        r = dict(zip((k for k, _ in SQUEUE_FIELDS), parts))
        if r["id"] in seen:  # squeue can list a job twice when it spans partitions
            continue
        seen.add(r["id"])
        jobs.append(Job(
            id=r["id"], user=r["user"], account=r["account"], name=r["name"], state=r["state"],
            nodes=_int(r["nodes"]), limit=parse_slurm_duration(r["limit"]),
            elapsed=parse_slurm_duration(r["elapsed"]) or 0, submit=parse_slurm_time(r["submit"]),
            start=parse_slurm_time(r["start"]), reason=r["reason"], partition=r["partition"],
            qos=r["qos"], priority=_int(r["priority"]), nodelist=r["nodelist"],
        ))
    return jobs


def parse_sinfo_nodes(text):
    """'frontier00001|allocated' lines -> {node: state}. Suffix flags (*, $, ~, #, ^) are kept."""
    states = {}
    for line in text.splitlines():
        if "|" in line:
            node, state = line.split("|", 1)
            states[node.strip()] = state.strip()
    return states


def parse_sinfo_partitions(text):
    parts = []
    for line in text.splitlines():
        f = line.split("|")
        if len(f) != 5:
            continue
        name, avail, total, fstr, limit = f
        a, i, o, t = (fstr.split("/") + ["0"] * 4)[:4]
        parts.append(Partition(name=name.rstrip("*"), default=name.endswith("*"), avail=avail,
                               total=_int(t or total), alloc=_int(a), idle=_int(i), other=_int(o),
                               timelimit=limit))
    return parts


def _num(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_showusage(text):
    """OLCF ``showusage`` table: project, allocation, usage, remaining | my usage."""
    allocs = []
    for line in text.splitlines():
        if "|" not in line or line.lstrip().startswith(("_", "Project")):
            continue
        cols = line.split("|")
        left = cols[0].split()
        if not left or not re.match(r"^[a-z]{3}\d", left[0]):
            continue
        mid = cols[1].split() if len(cols) > 1 else []
        mine = cols[2].split() if len(cols) > 2 else []
        allocs.append(Allocation(
            project=left[0], allocation=_num(left[1]) if len(left) > 1 else None,
            usage=_num(mid[0]) or 0.0 if mid else 0.0,
            remaining=_num(mid[1]) if len(mid) > 1 else None,
            my_usage=_num(mine[0]) or 0.0 if mine else 0.0,
        ))
    return allocs


def parse_sshare(text):
    out = []
    for line in text.splitlines():
        f = line.split("|")
        if len(f) < 8 or f[0] in ("Account", ""):
            continue
        try:
            out.append(FairShare(account=f[0].strip(), raw_usage=_int(f[4]), fairshare=float(f[7])))
        except ValueError:
            continue
    return out


SACCT_FIELDS = "JobID,JobName,Account,Partition,State,NNodes,ElapsedRaw,TimelimitRaw,Start,End,ExitCode"


def parse_sacct(text):
    jobs = []
    for line in text.splitlines():
        f = line.split("|")
        if len(f) != 11 or f[0] == "JobID":
            continue
        limit_min = _int(f[7], None)
        jobs.append(HistJob(
            id=f[0], name=f[1], account=f[2], partition=f[3], state=f[4].split()[0] if f[4] else "",
            nodes=_int(f[5]), elapsed=_int(f[6]), limit=limit_min * 60 if limit_min is not None else None,
            start=parse_slurm_time(f[8]), end=parse_slurm_time(f[9]), exit_code=f[10],
        ))
    jobs.sort(key=lambda j: j.start or datetime.min, reverse=True)
    return jobs


NODE_WATTS_MAX = 5000  # a Frontier node draws at most ~3 kW; some report glitches near 1e9


def parse_node_power(text):
    """
    ``scontrol show node -o`` -> {node: current watts} from Slurm's energy plugin. Impossible
    readings are left out, so a node can be missing; ``power_total`` fills those in.
    """
    out = {}
    for line in text.splitlines():
        m = re.match(r"NodeName=(\S+)", line)
        w = re.search(r"\bCurrentWatts=(\d+)", line)
        if m and w and int(w.group(1)) <= NODE_WATTS_MAX:
            out[m.group(1)] = int(w.group(1))
    return out


def power_total(watts, n_nodes=None):
    """Total watts, counting each node without a valid reading as the mean of the others."""
    if not watts:
        return None
    total = sum(watts.values())
    missing = max((n_nodes or 0) - len(watts), 0)
    return total + missing * total / len(watts)


def parse_partition_nodes(text):
    """``scontrol show partition -o`` -> {partition: [node names]}."""
    out = {}
    for line in text.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S*)", line))
        if "PartitionName" in kv:
            out[kv["PartitionName"]] = expand_hostlist(kv.get("Nodes", ""))
    return out


def parse_reservations(text):
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        kv = dict(re.findall(r"(\w+)=(\S*)", block))
        if "ReservationName" not in kv:
            continue
        out.append(Reservation(
            name=kv["ReservationName"], start=parse_slurm_time(kv.get("StartTime")),
            end=parse_slurm_time(kv.get("EndTime")), nodes=_int(kv.get("NodeCnt")),
            state=kv.get("State", ""), flags=kv.get("Flags", ""), users=kv.get("Users", ""),
        ))
    return out


# ---------------------------------------------------------------------------------------------
# command runner
# ---------------------------------------------------------------------------------------------


def run(cmd, timeout=30):
    """Run a command (list or string) and return stdout; raises RuntimeError with stderr on failure."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip() or f"{cmd[0]} exited {p.returncode}")
    return p.stdout


class Slurm:
    """The fetchers the console polls. Each returns parsed records; ``state_key`` names its slot."""

    def __init__(self, user=None, history_days=7):
        self.user = user or os.environ.get("USER", "")
        self.history_days = history_days

    def squeue(self):
        fmt = SEP.join(spec for _, spec in SQUEUE_FIELDS)
        return parse_squeue(run(["squeue", "-h", "-o", fmt]))

    def node_states(self):
        return parse_sinfo_nodes(run(["sinfo", "-h", "-N", "-o", "%N|%T"], timeout=60))

    def partitions(self):
        return parse_sinfo_partitions(run(["sinfo", "-h", "-o", "%P|%a|%D|%F|%l"]))

    def allocations(self):
        return parse_showusage(run(["showusage"], timeout=60))

    def fairshare(self):
        return parse_sshare(run(["sshare", "-U", "-P", "-n", "-u", self.user, "-o",
                                 "Account,User,RawShares,NormShares,RawUsage,NormUsage,EffectvUsage,FairShare"]))

    def history(self):
        return parse_sacct(run(["sacct", "-X", "-n", "-P", "-u", self.user, "-S", f"now-{self.history_days}days",
                                "-o", SACCT_FIELDS], timeout=60))

    def reservations(self):
        return parse_reservations(run(["scontrol", "show", "res"]))

    def node_power(self):
        return parse_node_power(run(["scontrol", "show", "node", "-o"], timeout=60))

    def partition_nodes(self):
        return parse_partition_nodes(run(["scontrol", "show", "partition", "-o"]))

    def job_detail(self, job_id):
        return run(["scontrol", "show", "job", str(job_id)])

    @staticmethod
    def cluster_name():
        try:
            out = run(["scontrol", "show", "config"], timeout=10)
            m = re.search(r"^ClusterName\s*=\s*(\S+)", out, re.M)
            return m.group(1) if m else ""
        except Exception:
            return ""
