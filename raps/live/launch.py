"""
Help getting a job running: where you may submit, what would start right now, and when a given
request would start.

- **Access**: Slurm's own settings say little (Frontier's partitions all read ``AllowAccounts=ALL``;
  membership is enforced by a site submit filter), so access is checked by asking Slurm:
  ``sbatch --test-only`` validates a 1-node job for each (project, partition) without submitting
  anything, and reports either Slurm's start estimate or the filter's error message.
- **Start right now**: from the RAPS twin's plan at time limits (how Slurm's backfill plans), the
  free nodes in each partition pool over time. A job of N nodes starts at once if N nodes stay
  free for its whole time limit, so the window for N is how long N nodes stay free.
- **Your job**: Slurm's ``--test-only`` estimate and the twin's estimate for a hypothetical
  job; the twin gives it a priority like recent jobs of the same project and size.
"""
import re
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime

from raps.live.slurm import Job, expand_hostlist, parse_slurm_duration, parse_slurm_time

SIZES = (1, 8, 32, 128, 512, 1024, 2048, 4096, 8192)
TEST_SECONDS = 600   # the 1-node job used for access checks


@dataclass
class PartitionInfo:
    name: str
    nodes: list
    max_time: int | None          # seconds
    default_time: int | None
    qos: str                      # partition QOS ("N/A" for none)
    allow_qos: list               # ["ALL"] or the QOS names allowed
    state: str
    default: bool = False
    job_qos: str | None = None              # QOS a job here runs under: the partition's, or one it allows
    max_nodes: int | None = None            # per job, from that QOS (MaxTRESPU / GrpTRES node=)
    max_wall: int | None = None             # min(partition MaxTime, QOS MaxWall)


def parse_partitions(text, qos_limits=None):
    """``scontrol show partition -o`` (+ ``qos_limits``) -> {name: PartitionInfo}."""
    out = {}
    for line in text.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S*)", line))
        if "PartitionName" not in kv:
            continue
        qos = kv.get("QoS", "N/A")
        allow = kv.get("AllowQos", "ALL").split(",")
        job_qos = qos if qos != "N/A" else (allow[0] if allow != ["ALL"] else None)
        lim = (qos_limits or {}).get(job_qos, (None, None, None))
        max_time = parse_slurm_duration(kv.get("MaxTime"))
        nodes_caps = [n for n in (lim[0], lim[2]) if n]
        walls = [w for w in (max_time, lim[1]) if w]
        out[kv["PartitionName"]] = PartitionInfo(
            name=kv["PartitionName"], nodes=expand_hostlist(kv.get("Nodes", "")),
            max_time=max_time, default_time=parse_slurm_duration(kv.get("DefaultTime")),
            qos=qos, allow_qos=allow, state=kv.get("State", ""), default=kv.get("Default") == "YES",
            job_qos=job_qos, max_nodes=min(nodes_caps) if nodes_caps else None, max_wall=min(walls) if walls else None)
    return out


def parse_qos_limits(text):
    """
    ``sacctmgr -n -P show qos format=Name,MaxTRESPU,MaxWall,GrpTRES`` ->
    {qos: (max nodes per user, max wall seconds, max nodes for the whole QOS)}.
    """
    out = {}
    for line in text.splitlines():
        f = line.split("|") + [""]
        if len(f) < 4:
            continue
        pu = re.search(r"node=(\d+)", f[1])
        grp = re.search(r"node=(\d+)", f[3])
        out[f[0]] = (int(pu.group(1)) if pu else None, parse_slurm_duration(f[2]) if f[2] else None,
                     int(grp.group(1)) if grp else None)
    return out


def parse_accounts(text):
    """``sacctmgr -n -P show assoc user=U format=Account,QOS`` -> {account: [qos]}."""
    out = {}
    for line in text.splitlines():
        f = line.split("|")
        if f and f[0]:
            out[f[0]] = [q for q in (f[1].split(",") if len(f) > 1 else []) if q]
    return out


@dataclass
class TestResult:
    account: str
    partition: str
    qos: str | None
    ok: bool
    start: datetime | None = None
    message: str = ""


def parse_test_only(account, partition, qos, returncode, text):
    m = re.search(r"to start at (\S+)", text)
    if returncode == 0 and m:
        return TestResult(account, partition, qos, True, parse_slurm_time(m.group(1)))
    # keep the filter's own explanation, without its boilerplate
    lines = [ln.replace("sbatch: error:", "").strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and "ERROR FOUND" not in ln and "not submitted" not in ln
             and ln != "allocation failure: Unspecified error"]
    return TestResult(account, partition, qos, False, None, " ".join(lines) or text.strip()[:200])


def test_only(account, partition, nodes=1, seconds=TEST_SECONDS, qos=None):
    """Ask Slurm whether (and when) a job would run. Nothing is submitted."""
    cmd = ["sbatch", "--test-only", "-A", account, "-p", partition, "-N", str(nodes),
           "-t", str(max(seconds // 60, 1)), "--wrap", "true"]
    if qos:
        cmd[2:2] = ["-q", qos]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return parse_test_only(account, partition, qos, p.returncode, (p.stderr or "") + (p.stdout or ""))


def qos_for(part, account_qos=None):
    """The -q a job here needs: None for the default, else an allowed QOS (service: develop/jupyter)."""
    if part.allow_qos == ["ALL"]:
        return None
    if part.job_qos in (account_qos or part.allow_qos):
        return part.job_qos
    return next((q for q in part.allow_qos if q in (account_qos or [])), part.allow_qos[0])


@dataclass
class Access:
    checked: datetime
    results: dict = field(default_factory=dict)   # (account, partition) -> TestResult
    default_account: str | None = None            # the user's Slurm default account

    def ok(self, partition, account=None):
        return any(r.ok for (a, p), r in self.results.items() if p == partition and account in (None, a))

    def reason(self, partition):
        msgs = [r.message for (a, p), r in self.results.items() if p == partition and not r.ok]
        return msgs[0] if msgs else ""


def check_access(slurm, layout, accounts, default_account=None):
    results = {}
    for a, qos_list in accounts.items():
        for name, part in layout.items():
            if part.state != "UP":
                continue
            results[(a, name)] = slurm.test_only(a, name, qos=qos_for(part, qos_list))
    return Access(datetime.now(), results, default_account)


# ---------------------------------------------------------------------------------------------
# start-right-now windows and what-if starts, from the twin
# ---------------------------------------------------------------------------------------------


def start_now_windows(plan, layout, sizes=SIZES):
    """
    {partition: [(nodes, seconds or None)]} from a twin run at time limits: how long N nodes stay
    free in the partition's pool (capped at the partition's max time); None if N are not free now.
    """
    out = {}
    for name, part in layout.items():
        pool = plan.pool_of_partition.get(name)
        if pool is None:
            continue
        free = plan.pool_free[pool]
        cap = part.max_wall
        rows = []
        for n in sizes:
            if part.max_nodes and n > part.max_nodes or not free or free[0][1] < n:
                rows.append((n, None))
                continue
            end = next((t for t, f in free if f < n), None)
            window = end if end is not None else float("inf")
            rows.append((n, min(window, cap) if cap else window))
        out[name] = rows
    return out


def base_priority(jobs, account, nodes, now):
    """
    Priority a new job would get: Frontier's age factor adds 1 per second waited
    (PriorityWeightAge 31536000 over PriorityMaxAge 365 days), so priority minus seconds waited
    is the rest (size, fair share, QOS). Use the median of that over pending jobs of the same
    project (else all), nearest in size.
    """
    pend = [j for j in jobs if j.pending and j.submit]
    same = [j for j in pend if j.account == account] or pend
    if not same:
        return 0
    same.sort(key=lambda j: abs(j.nodes - nodes))
    near = same[:max(len(same) // 3, 1)]
    base = sorted(j.priority - (now - j.submit).total_seconds() for j in near)
    return int(max(base[len(base) // 2], 0))


def what_if(state, partition_nodes, partition, account, nodes, seconds, now=None):
    """The twin's start estimate for a job you might submit now (None if not within its horizon)."""
    from raps.live.twin import simulate
    now = now or datetime.now()
    me = Job(id="what-if", user=state.user, account=account, name="(your job)", state="PENDING",
             nodes=nodes, limit=seconds, elapsed=0, submit=now, start=None, reason="Priority",
             partition=partition, qos="normal", priority=base_priority(state.jobs or [], account, nodes, now),
             nodelist="")
    res = simulate(replace(state, jobs=list(state.jobs or []) + [me]), partition_nodes, now=now)
    return res.starts.get("what-if")


def sbatch_line(account, partition, nodes, seconds, qos=None):
    h, rem = divmod(int(seconds), 3600)
    t = f"{h:02d}:{rem // 60:02d}:00"
    q = f" -q {qos}" if qos else ""
    return f"sbatch -A {account} -p {partition}{q} -N {nodes} -t {t} "


def parse_walltime(text):
    """'90' / '90m' (minutes), '2h', '1:30:00', '1-00:00:00' -> seconds."""
    text = text.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hm]?)", text)
    if m:
        v = float(m.group(1))
        return int(v * 3600) if m.group(2) == "h" else int(v * 60)
    return parse_slurm_duration(text)
