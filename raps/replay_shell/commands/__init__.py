"""The command set. ``build_registry`` is the one place that decides what the shell can run."""
from raps.replay_shell.registry import CommandRegistry


def build_registry():
    from raps.replay_shell.commands import fs, slurm
    registry = CommandRegistry()
    fs.register_all(registry)
    slurm.register_all(registry)
    return registry
