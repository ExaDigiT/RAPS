"""
`raps runs`: housekeeping for the per-run output directories (see raps.cache.runs_dir).
"""
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from raps.cache import runs_dir
from raps.utils import SubParsers

# Files every run writes even if the simulation never finished; a run with nothing else is a "stub"
STUB_FILES = {"sim_config.yaml", "snapshot.npz"}
DURATION_UNITS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_duration(text: str) -> float:
    """ Parse e.g. 12h, 30d, 2w into seconds """
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([mhdw])", text.strip())
    if not m:
        raise ValueError(f"Invalid duration '{text}', expected a number and one of m, h, d, w (e.g. 30d)")
    return float(m.group(1)) * DURATION_UNITS[m.group(2)]


@dataclass
class Run:
    path: Path
    mtime: float
    size: int
    is_stub: bool


def find_runs(root: Path) -> list[Run]:
    """ Run directories directly under root and under root/legacy, oldest first. """
    runs = []
    for parent in (root, root / "legacy"):
        if not parent.is_dir():
            continue
        for d in parent.iterdir():
            # Only directories RAPS wrote (they contain sim_config.yaml), never symlinks
            if d.is_symlink() or not d.is_dir() or not (d / "sim_config.yaml").exists():
                continue
            files = [f for f in d.rglob("*") if f.is_file()]
            runs.append(Run(
                path=d, mtime=d.stat().st_mtime, size=sum(f.stat().st_size for f in files),
                is_stub=all(f.relative_to(d).as_posix() in STUB_FILES or f.suffix == ".npz" for f in files),
            ))
    return sorted(runs, key=lambda r: r.mtime)


def select_runs(runs: list[Run], older_than: float | None, keep: int | None, stubs: bool, now: float) -> list[Run]:
    """ Runs matching every given filter. --keep protects the newest N runs from all filters. """
    protected = set(r.path for r in (runs[len(runs) - keep:] if keep else []))
    return [
        r for r in runs
        if r.path not in protected
        and (older_than is None or now - r.mtime > older_than)
        and (not stubs or r.is_stub)
    ]


def human_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def runs_add_parser(subparsers: SubParsers):
    parser = subparsers.add_parser("runs", description="Manage run output directories.")
    sub = parser.add_subparsers(required=True)

    prune = sub.add_parser("prune", description="""
        Delete old run output directories. Filters combine (a run must match all of them), and at
        least one is required. Nothing is deleted unless --yes is passed; otherwise the matching
        runs are only listed. Directories without a sim_config.yaml are never touched.
    """)
    prune.add_argument("--older-than", metavar="AGE", help="Runs last modified longer ago than this, e.g. 12h, 30d, 2w")
    prune.add_argument("--keep", type=int, metavar="N", help="Never delete the newest N runs")
    prune.add_argument("--stubs", action="store_true",
                       help="Only runs with no results, i.e. just sim_config.yaml and snapshot .npz files")
    prune.add_argument("--runs-dir", type=Path, help="Runs directory (default: $RAPS_RUNS_DIR or ./runs)")
    prune.add_argument("--yes", action="store_true", help="Actually delete instead of listing")
    prune.set_defaults(impl=runs_prune)


def runs_prune(args):
    if args.older_than is None and args.keep is None and not args.stubs:
        raise SystemExit("raps runs prune: pass at least one of --older-than, --keep, --stubs")
    try:
        older_than = parse_duration(args.older_than) if args.older_than else None
    except ValueError as e:
        raise SystemExit(f"raps runs prune: {e}")
    root = args.runs_dir or runs_dir()

    runs = find_runs(root)
    doomed = select_runs(runs, older_than, args.keep, args.stubs, time.time())
    for r in doomed:
        kind = "stub" if r.is_stub else "run "
        print(f"{kind}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(r.mtime))}  "
              f"{human_size(r.size):>9}  {r.path}")
    total = human_size(sum(r.size for r in doomed))
    if not args.yes:
        print(f"{len(doomed)} of {len(runs)} runs in {root} match ({total}). Dry run; pass --yes to delete.")
        return
    for r in doomed:
        shutil.rmtree(r.path)
    print(f"Deleted {len(doomed)} of {len(runs)} runs in {root} ({total}).")
