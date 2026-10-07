"""
Default locations for RAPS files, and the content-addressed cache of parsed replay snapshots.

Parsing raw telemetry with a dataloader is slow, so the result is stored as an .npz snapshot under
the cache directory, keyed on everything that can change the parsed jobs. Re-running the same
replay then loads the snapshot instead of re-running the dataloader.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

CACHE_VERSION = 1

# Simulation options that are passed to dataloaders and can change the jobs they return. Options
# that only affect the simulation (scheduler policy, cooling, UI, ...) are deliberately left out so
# they do not cause cache misses.
LOADER_KEYS = (
    "system", "partition", "dataloader", "start", "end", "time", "downscale", "time_unit",
    "jid", "validate", "encrypt", "filter", "reschedule", "live",
)


def cache_dir() -> Path:
    """ $RAPS_CACHE_DIR, else $XDG_CACHE_HOME/raps, else ~/.cache/raps """
    if os.environ.get("RAPS_CACHE_DIR"):
        return Path(os.environ["RAPS_CACHE_DIR"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "raps"


def snapshot_cache_dir() -> Path:
    return cache_dir() / "snapshots"


def runs_dir() -> Path:
    """ Parent of the per-run output directories: $RAPS_RUNS_DIR, else ./runs """
    return Path(os.environ.get("RAPS_RUNS_DIR") or "runs").expanduser()


def _fingerprint_path(path: Path) -> list:
    """ (relative name, size, mtime) of a file, or of every file under a directory. """
    path = path.resolve()
    if path.is_dir():
        entries = []
        for p in sorted(path.rglob("*")):
            if p.is_file():
                st = p.stat()
                entries.append((str(p.relative_to(path)), st.st_size, st.st_mtime_ns))
        return [str(path), entries]
    st = path.stat()
    return [str(path), st.st_size, st.st_mtime_ns]


def _source_hash(module) -> str:
    """ Hash of the dataloader's source (all .py files if it is a package), so edits invalidate. """
    h = hashlib.sha256()
    paths = [Path(p) for p in getattr(module, "__path__", [])]
    if paths:
        files = sorted(f for p in paths for f in p.rglob("*.py"))
    else:
        files = [Path(module.__file__)] if getattr(module, "__file__", None) else []
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()


def snapshot_cache_path(files: Iterable, dataloader, kwargs: dict) -> Path:
    """ Path of the cached snapshot for loading `files` with `dataloader` and these kwargs. """
    from raps import job as job_module  # Job layout changes invalidate old snapshots
    key = {
        "version": CACHE_VERSION,
        "files": [_fingerprint_path(Path(f)) for f in files],
        "loader": {k: kwargs.get(k) for k in LOADER_KEYS},
        "loader_source": _source_hash(dataloader),
        "job_source": _source_hash(job_module),
        "config": kwargs.get("config"),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    system = str(kwargs.get("system") or "unknown").replace("/", "_")
    return snapshot_cache_dir() / f"{system}-{digest}.npz"
