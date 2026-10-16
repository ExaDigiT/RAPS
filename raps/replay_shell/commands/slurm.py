"""
Simulated Slurm commands. They act on the simulation through ``ctx.adapter`` (a ``SimAdapter``); nothing
runs on the real machine and no real Slurm binary is called.

All of these are stubs for the intern (see README.md for the task list and recommended order):
``sbatch``, ``squeue``, ``sinfo``, ``scancel``, ``sacct``, ``srun``. The shell prints "not implemented yet"
when a handler raises ``NotImplementedError``. The pieces they build on already exist: ``jobscript.py``
(directive parsing), ``Engine.inject_job`` (thread-safe queueing) and the ``SimAdapter`` interface.
"""
from raps.replay_shell.registry import CommandError


def _sbatch(ctx, args):
    """sbatch [--runtime T] [-N n] [-t T] [-J name] script

    Read the script from the sandbox, take its ``#SBATCH`` directives, let options on the command line
    override them (as real sbatch does) and queue the job at the current simulated time.

    TODO(intern):
      1. Parse the arguments: options with values (``-N 4``, ``--nodes=4``) and one script name. Raise
         ``CommandError("usage: sbatch [options] script")`` without a script.
      2. ``spec = jobscript.parse_job_script(text)``, then ``jobscript.apply_option(spec, key, value)`` for each
         command-line option (it returns False for options it does not know; report those as errors).
         ``ValueError`` from the parser should become a ``CommandError``.
      3. Fall back to the script's file name for the job name; reject non-positive nodes or time limit.
      4. ``job_id = adapter.submit(spec)`` (see ``EngineAdapter.submit``), then print
         ``Submitted batch job <id>``.
      5. Later: --wrap, --test-only, and a warning for each directive in ``spec.ignored``.
    """
    ctx.need_sim()
    script = next((a for a in args if not a.startswith("-")), None)
    if script is None:
        raise CommandError("usage: sbatch [options] script")
    real = ctx.sandbox.resolve(script)   # keep this: it is the sandbox check for the script path
    if not real.is_file():
        raise CommandError(f"{script}: no such file")
    raise NotImplementedError("see the TODO in _sbatch")


def _squeue(ctx, args):
    """squeue: the simulated queue (running first, then pending).

    TODO(intern): ``adapter.queue()`` returns ``JobInfo`` records (adapter.py); print a Slurm-like table:
    JOBID NAME ACCOUNT ST NODES TIME TIME_LIMIT (``raps.utils.convert_seconds_to_hhmmss`` for the times).
    Then -u/--user, -j <id>, -t <states>, -o format strings, --start (expected start from the scheduler),
    sorting by priority, and the REASON column.
    """
    ctx.need_sim()
    raise NotImplementedError("see the TODO in _squeue")


def _sinfo(ctx, args):
    """sinfo: node counts by state for the simulated machine.

    TODO(intern): ``adapter.nodes()`` returns a ``NodeSummary`` (allocated, idle, down, total); print a
    table of states. Then per-partition rows, a node list (``-N``), ``-o`` formats, and a draining/reserved
    state.
    """
    ctx.need_sim()
    raise NotImplementedError("see the TODO in _sinfo")


def _scancel(ctx, args):
    """scancel <jobid> ...: cancel simulated jobs.

    TODO(intern): implement ``EngineAdapter.cancel`` (and ``Engine.cancel_job``, see the notes there);
    then this handler is nearly done: parse ids, call the adapter, report "scancel: Invalid job id N".
    Also -u USER, and cancelling a whole array.
    """
    raise NotImplementedError("see EngineAdapter.cancel")


def _sacct(ctx, args):
    """sacct: finished simulated jobs (id, name, state, elapsed, nodes).

    TODO(intern): implement ``EngineAdapter.finished`` and format the rows like sacct; add -j, -S/-E (in
    simulated time), --format, and ``--starttime`` handling.
    """
    raise NotImplementedError("see EngineAdapter.finished")


def _srun(ctx, args):
    """srun: an interactive step. In a simulation there is nothing to attach to, so the sensible meaning
    is "submit a job that runs this command for its time limit and report when it would start".

    TODO(intern): decide the semantics with the user first (see README.md, open questions).
    """
    raise NotImplementedError("semantics undecided")


def register_all(r):
    r.register("sbatch", "submit a job script to the simulated queue (not implemented)",
               "sbatch [-N n] [-t time] [-J name] [--runtime time] script")(_sbatch)
    r.register("squeue", "show the simulated queue (not implemented)", "squeue")(_squeue)
    r.register("sinfo", "show simulated node states (not implemented)", "sinfo")(_sinfo)
    r.register("scancel", "cancel a simulated job (not implemented)", "scancel jobid ...")(_scancel)
    r.register("sacct", "show finished simulated jobs (not implemented)", "sacct")(_sacct)
    r.register("srun", "run a command on the simulated machine (not implemented)", "srun [options] command")(_srun)
