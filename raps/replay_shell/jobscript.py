"""
Parse a batch script's ``#SBATCH`` directives into a ``JobSpec`` (what ``sbatch`` needs to build a RAPS job).

Only a few directives are understood so far. Unknown ones are collected in ``JobSpec.ignored`` so the
shell can warn instead of silently dropping them.

TODO(intern): more directives (see the table in README.md): --partition, --account, --dependency,
--array, --gres/--gpus-per-node, --exclusive, --begin, --nodelist/--exclude; and ``--nodes=MIN-MAX``.
"""
import re
import shlex
from dataclasses import dataclass, field


@dataclass
class JobSpec:
    name: str = "sbatch"
    nodes: int = 1
    time_limit: int = 3600          # seconds (Slurm's default depends on the partition)
    runtime: int | None = None      # seconds the simulated job really runs; None: use the time limit
    account: str = ""
    partition: str | None = None
    cpu_util: float = 0.5           # constant utilisation used as the job's trace (0..1)
    gpu_util: float = 0.5
    ignored: list = field(default_factory=list)   # directives we did not understand


def parse_time(text):
    """Slurm time (``M``, ``M:S``, ``H:M:S``, ``D-H``, ``D-H:M``, ``D-H:M:S``) to seconds."""
    text = text.strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
        parts = [int(p) for p in text.split(":")]
        h, m, s = (parts + [0, 0])[:3]   # with days the first field is hours
    else:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 3:
            h, m, s = parts
        else:
            raise ValueError(f"bad time: {text}")
    return ((days * 24 + h) * 60 + m) * 60 + s


_LONG = re.compile(r"^--([\w-]+)(?:=(.*))?$")
_SHORT = {"N": "nodes", "t": "time", "J": "job-name", "A": "account", "p": "partition"}


def apply_option(spec, key, value):
    """Set one option (long name without dashes) on the spec; False if it is not understood."""
    try:
        if key == "nodes":
            spec.nodes = int(value.split("-")[0])
        elif key == "time":
            spec.time_limit = parse_time(value)
        elif key == "job-name":
            spec.name = value
        elif key == "account":
            spec.account = value
        elif key == "partition":
            spec.partition = value
        elif key == "runtime":     # RAPS extension, not Slurm: how long the simulated job runs
            spec.runtime = parse_time(value)
        else:
            return False
    except ValueError:
        raise ValueError(f"bad value for --{key}: {value!r}")
    return True


def parse_job_script(text):
    """Read the ``#SBATCH`` (and ``#RAPS``) lines at the top of a script. Stops at the first command."""
    spec = JobSpec()
    for line in text.splitlines():
        line = line.strip()
        if not line or (line.startswith("#") and not line.startswith(("#SBATCH", "#RAPS"))):
            continue   # blank, shebang, comment
        if not line.startswith(("#SBATCH", "#RAPS")):
            break   # Slurm stops reading directives at the first executable line
        toks = shlex.split(line.split(None, 1)[1] if " " in line else "", comments=True)
        i = 0
        while i < len(toks):
            tok = toks[i]
            i += 1
            m = _LONG.match(tok)
            if m:
                key, value = m.group(1), m.group(2)
            elif tok[:1] == "-" and tok[1:2] in _SHORT:
                key, value = _SHORT[tok[1:2]], tok[2:].lstrip("=") or None
            else:
                spec.ignored.append(tok)
                continue
            if value is None and i < len(toks) and not toks[i].startswith("-"):
                value = toks[i]   # `-N 4` and `--nodes 4`
                i += 1
            if not apply_option(spec, key, value or ""):
                spec.ignored.append(tok)
    return spec
