"""
Background polling of Slurm, plus a small persisted time series of queue-wide numbers.

The poller runs in a daemon thread and refreshes each source on its own interval (``squeue`` is
cheap, ``sinfo -N`` takes seconds on Frontier). After every fetch it publishes a fresh
``ClusterState`` into a single "latest" slot that the UI polls, the same hand-off the RAPS
console uses for simulation snapshots.
"""
import json
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from raps.live.forecast import Forecaster
from raps.live.slurm import ClusterState, Slurm

# source -> (state field, Slurm method, interval seconds)
SOURCES = {
    "jobs": ("jobs", "squeue", 15),
    "nodes": ("node_states", "node_states", 60),
    "partitions": ("partitions", "partitions", 60),
    "history": ("history", "history", 120),
    "fairshare": ("fairshare", "fairshare", 600),
    "allocations": ("allocations", "allocations", 600),
    "reservations": ("reservations", "reservations", 300),
    "forecast": ("forecast", None, 600),  # Poller.forecaster.refresh; last, as the first pull is slow
}


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "raps-live"


class Trend:
    """Queue-wide numbers sampled after each squeue fetch, kept for `keep_hours` and saved to disk."""

    KEYS = ("run_jobs", "run_nodes", "pend_jobs", "pend_nodes", "backlog_nh", "my_run", "my_pend")

    def __init__(self, path=None, keep_hours=24):
        self.path = path
        self.keep = keep_hours * 3600
        self.samples = deque()
        if path is not None:
            self._load()

    def _load(self):
        try:
            cutoff = time.time() - self.keep
            with open(self.path) as f:
                for line in f:
                    try:
                        s = json.loads(line)
                    except ValueError:
                        continue
                    if s.get("t", 0) >= cutoff:
                        self.samples.append(s)
            # rewrite trimmed so the file does not grow without bound
            self._rewrite()
        except OSError:
            pass

    def _rewrite(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for s in self.samples:
                    f.write(json.dumps(s) + "\n")
            tmp.replace(self.path)
        except OSError:
            pass

    def add(self, jobs, user):
        run = [j for j in jobs if j.running]
        pend = [j for j in jobs if j.pending]
        s = {
            "t": time.time(),
            "run_jobs": len(run), "run_nodes": sum(j.nodes for j in run),
            "pend_jobs": len(pend), "pend_nodes": sum(j.nodes for j in pend),
            "backlog_nh": round(sum(j.node_hours for j in pend), 1),
            "my_run": sum(1 for j in run if j.user == user),
            "my_pend": sum(1 for j in pend if j.user == user),
        }
        self.samples.append(s)
        cutoff = s["t"] - self.keep
        while self.samples and self.samples[0]["t"] < cutoff:
            self.samples.popleft()
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(s) + "\n")
            except OSError:
                pass

    def series(self, key):
        return [s.get(key, 0) for s in self.samples]

    def span_seconds(self):
        return self.samples[-1]["t"] - self.samples[0]["t"] if len(self.samples) > 1 else 0


class Poller:
    def __init__(self, slurm: Slurm | None = None, trend: Trend | None = None, scale=1.0,
                 forecaster: Forecaster | None = None):
        self.slurm = slurm or Slurm()
        self.forecaster = forecaster  # None: no runtime forecasts
        self.trend = trend if trend is not None else Trend(cache_dir() / "trend.jsonl")
        self.scale = scale  # multiply every interval (e.g. 2.0 = poll half as often)
        self.state = ClusterState(user=self.slurm.user)
        self.busy = set()           # sources being fetched right now
        self._latest = self.state
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.sources = {k: v for k, v in SOURCES.items() if k != "forecast" or forecaster is not None}
        self._due = {name: 0.0 for name in self.sources}
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="raps-live-poller")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def latest(self):
        with self._lock:
            return self._latest

    def refresh(self, *names):
        """Fetch the named sources (all if none) as soon as possible."""
        for name in names or self.sources:
            if name in self._due:
                self._due[name] = 0.0
        self._wake.set()

    def _publish(self, **changes):
        with self._lock:
            self.state = replace(self.state, **changes)
            self._latest = self.state

    def _loop(self):
        if not self.state.cluster:
            self._publish(cluster=self.slurm.cluster_name())
        while not self._stop.is_set():
            now = time.monotonic()
            # fetch in a stable order with the fast, important sources first
            for name, (attr, method, interval) in self.sources.items():
                if self._stop.is_set():
                    return
                if self._due[name] > now:
                    continue
                self.busy.add(name)
                try:
                    fetch = self.forecaster.refresh if name == "forecast" else getattr(self.slurm, method)
                    value = fetch()
                    errors = {k: v for k, v in self.state.errors.items() if k != name}
                    self._publish(**{attr: value}, updated={**self.state.updated, name: datetime.now()},
                                  errors=errors)
                    if name == "jobs":
                        self.trend.add(value, self.slurm.user)
                except Exception as exc:  # keep the last good data; report the failure
                    msg = str(exc).splitlines()[0] if str(exc) else traceback.format_exc(limit=1)
                    self._publish(errors={**self.state.errors, name: msg})
                finally:
                    self.busy.discard(name)
                self._due[name] = time.monotonic() + interval * self.scale
            wait = max(min(self._due.values()) - time.monotonic(), 0.2)
            self._wake.wait(wait)
            self._wake.clear()
