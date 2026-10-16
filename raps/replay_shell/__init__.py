"""
Interactive shell for the simulation console (``raps run --shell``): Slurm-like commands that act on the
*simulated* machine, plus a sandbox directory for job scripts. Nothing here runs on the real system.

See README.md in this directory for the design, what is done and what is left to do.

- ``sandbox``: a real directory the shell is confined to (``Sandbox.resolve`` is the only path gate)
- ``registry``: ``Command``/``CommandRegistry`` and ``ReplayShell`` (parse a line, dispatch, redirection)
- ``adapter``: ``SimAdapter``, the narrow interface commands use to talk to the engine (``EngineAdapter``)
- ``jobscript``: parse ``#SBATCH`` directives into a ``JobSpec``
- ``commands``: the built-ins (``fs``) and the simulated Slurm commands (``slurm``)
- ``view``: the Textual tab (``ShellView``) that hosts a ``ReplayShell`` inside ``RapsApp``
"""
from raps.replay_shell.adapter import EngineAdapter, SimAdapter
from raps.replay_shell.registry import Command, CommandRegistry, ReplayShell
from raps.replay_shell.sandbox import Sandbox, SandboxError

__all__ = ["Sandbox", "SandboxError", "Command", "CommandRegistry", "ReplayShell", "SimAdapter", "EngineAdapter",
           "default_sandbox_dir", "build_shell", "seed_sandbox"]


def default_sandbox_dir():
    """~/.raps/sandbox, created on first use."""
    from pathlib import Path
    return Path.home() / ".raps" / "sandbox"


def build_shell(engine=None, sandbox_dir=None):
    """A ready-to-use ``ReplayShell``. ``engine`` may be None (filesystem commands only)."""
    from raps.replay_shell.commands import build_registry
    sandbox = Sandbox(sandbox_dir or default_sandbox_dir(), create=True)
    seed_sandbox(sandbox)
    return ReplayShell(sandbox, EngineAdapter(engine) if engine is not None else None, build_registry())


DEFAULT_JOB_SCRIPT = """#!/bin/bash
#SBATCH -J demo
#SBATCH -N 4
#SBATCH -t 00:30:00
#RAPS --runtime 00:10:00

# Example job for the simulated shell: try `sbatch job.sh`, then `squeue`.
# #SBATCH lines must come before the first command. The commands below are never run;
# the simulated job lasts --runtime (default: the -t limit).
srun ./my_app
"""


def seed_sandbox(sandbox):
    """Put an example ``job.sh`` in the sandbox if there is none (an edited one is never overwritten)."""
    path = sandbox.root / "job.sh"
    if not path.exists():
        path.write_text(DEFAULT_JOB_SCRIPT)
