from argparse import Namespace
from datetime import datetime
from pathlib import Path

from .loader import load_data
from .download import download as _download_from_args
from .utils import DEFAULT_START, DEFAULT_END

__all__ = ["load_data", "download"]


def download(dest: Path, start: datetime | None, end: datetime | None):
    """Bridges the generic `raps download` CLI to mit_supercloud's S3 downloader,
    which was written against its own standalone cli.py args namespace.

    Partition (part-cpu/part-gpu/all) is inferred from dest's last path component,
    since `raps download --system mit_supercloud[/part-cpu|/part-gpu]` puts the
    system name there (see run_download in raps/telemetry.py).
    """
    partition = dest.name if dest.name in ("part-cpu", "part-gpu") else "all"
    args = Namespace(
        start=start.isoformat() if start else DEFAULT_START,
        end=end.isoformat() if end else DEFAULT_END,
        partition=partition,
        outdir=str(dest),
        bucket="mit-supercloud-dataset",
        prefix="datacenter-challenge/202201/",
        max_jobs=None,
        dry_run=False,
    )
    return _download_from_args(args)
