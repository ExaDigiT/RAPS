#!/usr/bin/env python
"""
A scripted tour of `raps live` on synthetic data (no Slurm, nothing real on screen), for recording.

    python scripts/demo_live.py --play                     run the tour in this terminal
    python scripts/demo_live.py --record demo.cast         record it with asciinema (needs `pip install asciinema`)
    python scripts/demo_live.py --record real.cast --real  the same tour on the real machine (Slurm); real user,
                                                           project and job names are on screen

The synthetic machine is Frontier-sized (9,408 nodes, ~93 % busy) with made-up users, projects and job
names. A driver thread types the keys and shows a caption (a toast) for each feature.
--record runs the tour inside a pseudo-terminal of --cols x --rows, so it works from a script or over ssh.
"""
import argparse
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raps.live.slurm import (  # noqa: E402
    SEP, Allocation, FairShare, HistJob, Job, Partition, Reservation, Slurm, parse_partition_nodes,
)

N_NODES = 9408
USER = "demo"
PROJECTS = ["ast101", "bio203", "cfd305", "chm410", "csc512", "fus606", "geo708", "lgt810", "mat912", "nph114",
            "phy216", "stf318", "ven420", "cli522"]
NAMES = ["lammps_run", "qmc_scan", "climate_ens", "cfd_les", "md_prod", "train_llm", "fft_bench", "dft_relax",
         "amr_sim", "hmc_gauge", "seismic_fwi", "bse_calc"]
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SIZE_W = [14, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1]
LIMITS = [1800, 3600, 7200, 14400, 21600, 43200, 86400]
LIMIT_W = [4, 8, 8, 8, 6, 4, 2]

PARTITIONS = "\n".join([
    f"PartitionName=batch AllowQos=ALL Default=YES QoS=N/A DefaultTime=01:00:00 MaxTime=2-00:00:00 "
    f"Nodes=frontier[00001-{N_NODES:05d}] State=UP TotalNodes={N_NODES}",
    f"PartitionName=extended AllowQos=ALL Default=NO QoS=N/A DefaultTime=01:00:00 MaxTime=7-00:00:00 "
    f"Nodes=frontier[00001-{N_NODES - 400:05d}] State=UP TotalNodes={N_NODES - 400}",
])


def hostlist(idx):
    """[1, 2, 3, 7] -> frontier[00001-00003,00007]"""
    idx = sorted(idx)
    parts, lo, prev = [], idx[0], idx[0]
    for i in idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(f"{lo:05d}" if lo == prev else f"{lo:05d}-{prev:05d}")
        if i is not None:
            lo = prev = i
    return f"frontier{parts[0]}" if len(parts) == 1 and "-" not in parts[0] else f"frontier[{','.join(parts)}]"


class DemoSlurm(Slurm):
    """A made-up cluster, generated once with a fixed seed."""

    def __init__(self, seed=7):
        super().__init__(user=USER)
        self.rng = rng = random.Random(seed)
        self.now = now = datetime.now()
        self.user_proj = {f"user{i:02d}": rng.choice(PROJECTS) for i in range(1, 41)}
        self.user_proj[USER] = "stf318"
        self.jobs, self.watts = [], {}
        used = bytearray(N_NODES + 1)
        down = set(rng.sample(range(1, N_NODES + 1), 60))
        for n in down:
            used[n] = 2
        self.down = down
        self._jid = 4_000_000

        free = [n for n in range(1, N_NODES + 1) if not used[n]]

        def place(size):
            """Scattered placement, as on a busy Frontier: any `size` of the free nodes; None if too few are free."""
            if size > len(free):
                return None
            picked = sorted(rng.sample(free, size))
            gone = set(picked)
            free[:] = [n for n in free if n not in gone]
            for n in picked:
                used[n] = 1
            return picked

        def job(user, account, name, size, limit, state, reason, nodelist="", elapsed=0, prio=0, age=3600, part="batch"):
            self._jid += rng.randint(1, 9)
            return Job(id=str(self._jid), user=user, account=account, name=name, state=state, nodes=size, limit=limit,
                       elapsed=elapsed, submit=now - timedelta(seconds=age + elapsed),
                       start=now - timedelta(seconds=elapsed) if state == "RUNNING" else None, reason=reason,
                       partition=part, qos="normal", priority=prio, nodelist=nodelist)

        def run(user, name, size, limit=None, frac=None, part="batch"):
            limit = limit or rng.choices(LIMITS, LIMIT_W)[0]
            nodes = place(size)
            if nodes is None:
                return
            frac = frac if frac is not None else rng.uniform(0.04, 0.9)
            self.jobs.append(job(user, self.user_proj[user], name, size, limit, "RUNNING", "None", hostlist(nodes),
                                 int(limit * frac), part=part, age=rng.randint(60, 7200)))
            w = rng.uniform(900, 2500)
            for n in nodes:
                self.watts[f"frontier{n:05d}"] = int(w * rng.uniform(0.94, 1.06))

        # my own jobs first, so they are on the map
        run(USER, "md_prod", 64, 21600, 0.35)
        run(USER, "dft_relax", 8, 7200, 0.6)
        run(USER, "fft_bench", 1, 3600, 0.2)
        others = [u for u in self.user_proj if u != USER]
        busy_target = int(0.93 * (N_NODES - len(down)))
        while sum(j.nodes for j in self.jobs) < busy_target:
            size = rng.choices(SIZES, SIZE_W)[0]
            before = len(self.jobs)
            run(rng.choice(others), rng.choice(NAMES), size)
            if len(self.jobs) == before and size > 16:
                run(rng.choice(others), rng.choice(NAMES), rng.choice([1, 2, 4, 8]))
        for n in range(1, N_NODES + 1):
            if used[n] == 0:
                self.watts[f"frontier{n:05d}"] = rng.randint(450, 520)
            elif used[n] == 2:
                self.watts[f"frontier{n:05d}"] = rng.randint(80, 140)
        self.used = used
        # the pending queue
        pend = []
        for i in range(520):
            user = rng.choice(others)
            size = rng.choices(SIZES, SIZE_W)[0]
            reason = rng.choices(["Priority", "Resources", "Dependency", "JobHeldUser"], [60, 25, 10, 5])[0]
            pend.append(job(user, self.user_proj[user], rng.choice(NAMES), size, rng.choices(LIMITS, LIMIT_W)[0],
                            "PENDING", reason, prio=rng.randint(50_000, 900_000), age=rng.randint(60, 70_000),
                            part=rng.choice(["batch", "batch", "batch", "extended"])))
        for size, name, prio, acct in ((128, "md_prod", 410_000, "stf318"), (512, "train_llm", 380_000, "csc512"),
                                       (16, "dft_relax", 120_000, "stf318"), (4, "fft_bench", 90_000, "stf318")):
            pend.append(job(USER, acct, name, size, 14400, "PENDING", "Priority", prio=prio, age=5400))
        for j in sorted(pend, key=lambda j: -j.priority)[:25]:   # Slurm's backfill estimate for the head of the queue
            j.start = now + timedelta(seconds=rng.randint(600, 36000))
        self.jobs += pend
        # finished-run history: every user's runs, so the forecasts have something to go on
        self.runs = []
        for user, proj in self.user_proj.items():
            for name in rng.sample(NAMES, 5):
                for _ in range(rng.randint(3, 8)):
                    limit_min = rng.choices(LIMITS, LIMIT_W)[0] // 60
                    elapsed = int(limit_min * 60 * rng.uniform(0.25, 0.95))
                    nodes = rng.choices(SIZES, SIZE_W)[0]
                    self.runs.append(SEP.join([str(rng.randint(1, 10 ** 7)), user, proj, name, str(nodes),
                                                  str(limit_min), str(elapsed),
                                                  (now - timedelta(hours=rng.randint(1, 600))).isoformat(timespec="seconds"),
                                                  str(int(elapsed * nodes * rng.uniform(1000, 2200)))]))

    # -- the fetchers the poller calls -------------------------------------------------------------
    def squeue(self):
        # elapsed time advances while the demo runs
        age = (datetime.now() - self.now).total_seconds()
        return [Job(**{**j.__dict__, "elapsed": j.elapsed + int(age)}) if j.state == "RUNNING" else j
                for j in self.jobs]

    def node_states(self):
        out = {}
        for n in range(1, N_NODES + 1):
            out[f"frontier{n:05d}"] = {0: "idle", 1: "allocated", 2: "drained"}[self.used[n]]
        return out

    def node_power(self):
        return dict(self.watts)

    def partitions(self):
        busy = sum(1 for n in range(1, N_NODES + 1) if self.used[n] == 1)
        out = len(self.down)
        return [Partition("batch*", True, "up", N_NODES, busy, N_NODES - busy - out, out, "2-00:00:00"),
                Partition("extended", False, "up", N_NODES - 400, busy, N_NODES - 400 - busy - out, out, "7-00:00:00")]

    def allocations(self):
        return [Allocation("stf318", 120_000, 61_500, 58_500, 9_200), Allocation("csc512", 300_000, 288_000, 12_000, 1_400)]

    def fairshare(self):
        return [FairShare("stf318", 61_500, 0.62), FairShare("csc512", 288_000, 0.18)]

    def history(self):
        now, rng = datetime.now(), random.Random(3)
        out = []
        for i in range(14):
            lim = rng.choice([3600, 7200, 14400])
            state = rng.choices(["COMPLETED", "TIMEOUT", "FAILED", "CANCELLED"], [10, 2, 1, 1])[0]
            el = lim if state == "TIMEOUT" else int(lim * rng.uniform(0.2, 0.95))
            end = now - timedelta(hours=3 * i + 1)
            out.append(HistJob(str(3_900_000 + i), rng.choice(["md_prod", "dft_relax", "fft_bench"]), "stf318", "batch",
                               state, rng.choice([1, 4, 8, 64]), el, lim, end - timedelta(seconds=el), end,
                               "0:0" if state == "COMPLETED" else "1:0"))
        return out

    def reservations(self):
        now = datetime.now()
        return [Reservation("maint_pm", now + timedelta(days=2, hours=3), now + timedelta(days=2, hours=15), 9408,
                            "INACTIVE", "MAINT,IGNORE_JOBS", "root"),
                Reservation("demo_day", now - timedelta(hours=1), now + timedelta(hours=5), 256, "ACTIVE", "", "user05")]

    def layout(self):
        from raps.live.launch import parse_partitions, parse_qos_limits
        return parse_partitions(PARTITIONS, parse_qos_limits(""))

    def accounts(self):
        return {"stf318": [], "csc512": []}

    def default_account(self):
        return "stf318"

    def test_only(self, account, partition, nodes=1, seconds=600, qos=None):
        from raps.live.launch import parse_test_only
        if account == "csc512" and partition == "extended":
            return parse_test_only(account, partition, qos, 1, "sbatch: error: Account csc512 may not use partition "
                                   "extended.")
        start = datetime.now() + timedelta(seconds=min(nodes * 7, 40000))
        return parse_test_only(account, partition, qos, 0, f"sbatch: Job 77 to start at {start:%Y-%m-%dT%H:%M:%S} "
                               f"using {nodes} processors on nodes frontier00003 in partition {partition}")

    def partition_nodes(self):
        return parse_partition_nodes(PARTITIONS)

    def job_detail(self, job_id):
        j = next((j for j in self.jobs if j.id == str(job_id)), None)
        if j is None:
            return f"slurm_load_jobs error: Invalid job id specified: {job_id}"
        return (f"JobId={j.id} JobName={j.name}\n   UserId={j.user}(1000) GroupId={j.account}(1000)\n"
                f"   Priority={j.priority} Nice=0 Account={j.account} QOS={j.qos}\n"
                f"   JobState={j.state} Reason={j.reason} Dependency=(null)\n"
                f"   NumNodes={j.nodes} Partition={j.partition} NodeList={j.nodelist}\n"
                f"   TimeLimit={j.limit // 3600:02d}:{j.limit % 3600 // 60:02d}:00")

    @staticmethod
    def cluster_name():
        return "frontier"


def app_ff(app):
    return getattr(app, "ff", None)


# ---------------------------------------------------------------------------------------------------------
# the tour
# ---------------------------------------------------------------------------------------------------------

def make_real_app(args):
    """The console on the real machine. A scratch cache dir (with a copy of the queue trend) keeps the recording
    out of the shell history and the trend file."""
    import shutil
    import tempfile
    real_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "raps-live"
    scratch = Path(tempfile.mkdtemp(prefix="raps-live-demo-cache-"))
    (scratch / "raps-live").mkdir()
    if (real_cache / "trend.jsonl").exists():
        shutil.copy(real_cache / "trend.jsonl", scratch / "raps-live" / "trend.jsonl")
    os.environ["XDG_CACHE_HOME"] = str(scratch)
    from raps.live import app as live_app
    from raps.live.forecast import Forecaster
    from raps.live.poller import Poller
    from raps.live.widgets import set_nodes_per_cab
    cluster = Slurm.cluster_name()
    system = cluster if cluster and (live_app.CONFIG_DIR / f"{cluster}.yaml").exists() else "frontier"
    set_nodes_per_cab(live_app.nodes_per_rack(system))
    poller = Poller(Slurm(history_days=7), forecaster=Forecaster(days=30), twin=True, system=system, access=True)
    return live_app.LiveApp(poller)


def make_app(args):
    if args.real:
        return make_real_app(args)
    from raps.live.app import LiveApp
    from raps.live.forecast import Forecaster, parse_runs
    from raps.live.poller import Poller, Trend
    from raps.live.widgets import set_nodes_per_cab
    from raps.live.app import nodes_per_rack
    slurm = DemoSlurm()
    set_nodes_per_cab(nodes_per_rack("frontier"))
    forecaster = Forecaster(days=30, fetch=lambda since: parse_runs("\n".join(slurm.runs)))
    # a made-up history of the queue, so the sparklines in the tabs have something to show
    trend = Trend(None)
    rng = random.Random(1)
    t0 = time.time()
    for k in range(240):
        s = {"t": t0 - (240 - k) * 120, "run_jobs": 30 + int(12 * rng.random()), "run_nodes": 8500 + int(300 * rng.random()),
             "pend_jobs": 480 + k // 4, "pend_nodes": 60000 + 90 * k, "backlog_nh": 300000 + 900 * k,
             "my_run": 3, "my_pend": 4, "power_mw": round(19 + 4 * rng.random(), 3)}
        trend.samples.append(s)
    poller = Poller(slurm, trend=trend, forecaster=forecaster, twin=True, system="frontier", access=True)
    return LiveApp(poller)


class Driver(threading.Thread):
    """Types keys into the app and shows a caption for each step."""

    def __init__(self, app, speed=1.0, real=False):
        super().__init__(daemon=True)
        self.app, self.speed, self.real = app, speed, real

    def wait(self, s):
        time.sleep(s / self.speed)

    def say(self, text, secs=4):
        self.app.call_from_thread(self.app.notify, text, title="raps live", timeout=secs)

    def key(self, *keys, gap=0.35):
        from textual import events
        from textual.keys import _character_to_key
        for k in keys:
            if type(self.app.screen).__name__ == "ConfirmScreen":   # never confirm a cancel/hold on a real job
                self.app.post_message(events.Key("escape", None))
                self.wait(0.5)
            if len(k) == 1:
                ev = events.Key(_character_to_key(k), k)
            else:
                ev = events.Key(k, "\r" if k == "enter" else None)
            self.app.post_message(ev)
            self.wait(gap)

    def focus(self, selector, value=None):
        """Focus a widget (and set an Input's text): digits and tab would otherwise switch tabs."""
        from textual.widgets import Input

        def go():
            w = self.app.query_one(selector)
            if value is not None and isinstance(w, Input):
                w.value = ""
            w.focus()
        self.app.call_from_thread(go)
        self.wait(0.3)

    def type(self, text, gap=0.07):
        self.key(*text, gap=gap)

    def run(self):
        app, k = self.app, self.key
        try:
            t0 = time.time()
            while time.time() - t0 < 240:  # first data in, twin and forecasts done
                st = app.poller.latest()
                if st.twin is not None and st.forecast is not None and st.access is not None:
                    break
                time.sleep(0.2)
            self.wait(1.5)
            self.tour(k)
        finally:
            time.sleep(1.0)
            app.call_from_thread(app.exit)

    def hottest_node(self):
        """The node drawing the most power among those running a job: the reddest square on the power map."""
        from raps.live.analysis import node_job_map
        st = self.app.poller.latest()
        busy = node_job_map(st.jobs)
        hot = [(w, n) for n, w in (st.node_power or {}).items() if n in busy]
        return max(hot)[1] if hot else None

    def move_cursor(self, name):
        """Put the Nodes tab's white cursor on a node (the arrow keys cannot get across 9,000 nodes in a demo)."""
        def go():
            nm = self.app.query_one("#nd-map")
            nm.cursor = name
            nm.refresh_image()
            nm.post_message(nm.Hover([name]))
        self.app.call_from_thread(go)

    def popular_name(self):
        """A job name worth filtering the Queue on: the most common one among the pending jobs."""
        from collections import Counter
        jobs = self.app.poller.latest().jobs or []
        names = Counter(j.name for j in jobs if j.pending and len(j.name) >= 4)
        return (names.most_common(1)[0][0] if names else "job")[:6]

    def tour(self, k):
        if self.real:
            self.say("Frontier right now: everything on these screens is live from Slurm", 6)
            self.wait(2)
        # ---- Overview
        self.say("Overview: utilization, power, the node map and the RAPS twin's next 24 h", 6)
        self.wait(5)
        self.say("c: color the node map: state, job, project, my jobs, power", 4)
        for _ in range(5):
            k("c")
            self.wait(1.6)
        k("c")
        self.wait(1)
        # ---- My Jobs
        self.say("2  My Jobs: queue position, start forecasts, history and allocations", 5)
        k("2")
        self.wait(4)
        if self.app.poller.latest().my_jobs:
            k("down", "down")
            self.wait(0.5)
            k("enter")
            self.say("enter: the job's details, and where it runs", 4)
            self.wait(3.5)
            k("escape")
        self.wait(0.8)
        # ---- Queue
        self.say("3  Queue: every job, with forecast runtime, power and energy", 5)
        k("3")
        self.wait(3)
        self.say("e: sort by energy   s: other sorts", 3)
        k("e")
        self.wait(2.5)
        k("s", "s")
        self.wait(2)
        self.say("/: filter on any text", 3)
        k("slash")
        self.type(self.popular_name())
        self.wait(2.5)
        k("escape")
        self.wait(1)
        self.say("m: only my jobs    f: running / pending / all", 4)
        k("m")
        self.wait(2)
        k("m")
        k("f")
        self.wait(1.5)
        k("f")
        self.wait(1.5)
        # ---- Nodes
        self.say("4  Nodes: a cabinet-by-cabinet map; arrows move the white cursor, the line above says what is under it", 6)
        k("4")
        self.wait(2)
        for key in ["right"] * 6 + ["down"] * 3 + ["right_square_bracket"] * 3:
            k(key, gap=0.25)
        self.wait(1.5)
        self.say("c: color by power, state, job, project, mine", 3)
        k("c", "c", "c", "c", "c", gap=1.4)   # a full cycle, back to power
        self.wait(1)
        hot = self.hottest_node()
        if hot:
            self.say("Power view: red is a node near its maximum draw. Jump to the hottest one", 5)
            self.wait(1)
            self.move_cursor(hot)
            self.wait(3)
            self.say("enter (or a click) opens the job running on that node", 5)
            k("enter")
            self.wait(7)
            k("escape")
        self.wait(1)
        k("escape")
        # ---- System
        self.say("5  System: partitions, reservations, why jobs wait, top projects and users", 5)
        k("5")
        self.wait(5)
        # ---- Launch
        self.say("7  Launch: where, and when, could my next job start? (nothing is submitted)", 6)
        k("7")
        self.wait(2)
        self.focus("#ln-nodes", "")
        self.type("256")
        self.focus("#ln-time", "")
        self.type("6h")
        self.focus("#ln-table")
        k("t")
        self.wait(6)
        # ---- Shell
        self.say("6  Shell: run anything without leaving the console", 4)
        k("6")
        self.wait(1)
        self.type("sinfo -s" if self.real else "echo any command works here")
        k("enter")
        self.wait(4 if self.real else 2.5)
        k("escape")
        # ---- export x
        self.say("x: write the queue as a RAPS replay (jobs.npz + experiment.yaml) and stage `raps run`", 6)
        k("x")
        self.wait(7)
        k("escape")   # leave the staged prompt: the keys that follow switch tabs again
        self.wait(0.5)
        # ---- fast-forward (starts at 30 min of machine time per second; the forecast is about 27 h long)
        k("1")
        self.wait(1)
        self.say("ff  (f twice): fast-forward. The twin's forecast plays on the same tabs", 6)
        k("f", gap=0.2)
        k("f")
        if self.real:   # the real forecast is days long: pick a speed that plays it in about 50 s
            from raps.live.fastforward import SPEEDS
            for _ in range(300):
                if app_ff(self.app) is not None:
                    break
                time.sleep(0.2)
            end = app_ff(self.app).end if app_ff(self.app) is not None else 86400
            idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] * 50 - end))
            self.app.ff_speed = idx
        self.wait(7)
        self.say("l faster, j slower, k pause, 0 reset speed (the keys of raps run)", 5)
        k("l")
        self.wait(3)
        k("j")
        self.wait(3)
        self.say("Jobs finish and start, nodes change hands, power follows", 4)
        self.wait(5)
        self.say("Every tab follows the clock: Queue", 4)
        k("3")
        self.wait(6)
        self.say("Nodes", 2)
        k("4")
        self.wait(5)
        self.say("My Jobs: when do mine start?", 4)
        k("2")
        self.wait(5)
        k("1")
        self.wait(3)
        self.say("k: pause", 2)
        k("k")
        self.wait(3)
        k("k")
        self.say("<<  restart from now", 3)
        self.wait(1)
        k("less_than_sign", gap=0.15)
        k("less_than_sign")
        self.wait(5)
        k("escape")
        self.say("esc: back to LIVE", 3)
        self.wait(3)
        # ---- help
        self.say("?  everything is listed in the help", 3)
        k("question_mark")
        self.wait(5)
        k("escape")
        self.wait(1)
        self.say("raps live: the machine now, and its forecast, in one console. q quits.", 4)
        self.wait(3)
        k("q")


def play(args):
    os.environ.setdefault("RAPS_LIVE_RUNS", str(Path(os.environ.get("TMPDIR", "/tmp")) / "raps-live-demo-runs"))
    app = make_app(args)
    Driver(app, args.speed, args.real).start()
    app.run()


def record(args):
    """Run the tour in a pseudo-terminal of the requested size, recorded by asciinema."""
    import fcntl
    import pty
    import select
    import struct
    import termios
    out = str(Path(args.record).resolve())
    cmd = ["asciinema", "rec", "--overwrite", "-i", "2", "-c",
           f"{sys.executable} {Path(__file__).resolve()} --play --speed {args.speed}{' --real' if args.real else ''}", out]
    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(TERM="xterm-256color", COLORTERM="truecolor", COLUMNS=str(args.cols), LINES=str(args.rows))
        os.execvp(cmd[0], cmd)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", args.rows, args.cols, 0, 0))
    while True:
        try:
            r, _, _ = select.select([fd], [], [], 1.0)
            if r and not os.read(fd, 65536):
                break
        except OSError:
            break
    os.waitpid(pid, 0)
    squeeze_loading(out)
    print(f"recorded {out}")


def squeeze_loading(path, keep=4.0):
    """Squeeze the time before the first caption (Slurm and forecasts loading) into `keep` seconds,
    keeping every frame, so a real-data recording does not open with a minute of spinner."""
    import json
    lines = Path(path).read_text().splitlines()
    ev = [json.loads(ln) for ln in lines[1:]]
    first = next((e[0] for e in ev if "raps live" in e[2] and "\u258c" in e[2]), None)  # the toast's left bar
    if first is None or first <= keep + 2:
        return
    span = first - 1.0   # everything before 1 s ahead of the first caption is loading
    for e in ev:
        e[0] = e[0] * keep / span if e[0] < span else e[0] - span + keep
    out = [lines[0]] + [json.dumps(e, ensure_ascii=False) for e in ev]
    Path(path).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play", action="store_true", help="run the tour in this terminal")
    ap.add_argument("--record", metavar="FILE.cast", help="record the tour with asciinema")
    ap.add_argument("--speed", type=float, default=1.0, help="play the tour faster (2 = twice as fast)")
    ap.add_argument("--real", action="store_true", help="use the real machine (Slurm) instead of synthetic data")
    ap.add_argument("--cols", type=int, default=150)
    ap.add_argument("--rows", type=int, default=42)
    a = ap.parse_args()
    if a.record:
        record(a)
    elif a.play:
        play(a)
    else:
        ap.print_help()
